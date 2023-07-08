

import dataclasses
from typing import ClassVar, Dict

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import evaluate
import numpy as np
from transformers import pipeline
import comet_ml
from comet_ml import Experiment
from sklearn.metrics import f1_score
import torch
from torch.optim import AdamW
import torch
from nlpaf.util import helpers
from torch.utils.data.dataloader import DataLoader
from evaluate import EvaluationModule
import os
from tqdm import tqdm
import hashlib
from accelerate.logging import get_logger
import datasets
import logging
import shutil 
import transformers
from glob import glob

@dataclasses.dataclass
class TextClassificationWAccelerate:
    dataset: 'DatasetDict | str'
    id2label: Dict  # = dataclasses.field(default_factory=dict)
    label2id: Dict  # = dataclasses.field(default_factory=dict)
    training_key: str
    eval_key: str
    undersample: bool = False

    text_col: str = "text"

    pretrained_model_name: str = "microsoft/deberta-base"
    metric: str = "f1"
    averaged_metric: str = "macro"
    model_org: str = "notaphoenix"
    uncase: bool = False

    # Training specific
    output_dir: str = "my_awesome_model"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch" # epoch or a number or None
    load_best_model_at_end: bool = True
    push_to_hub: bool = True
    hub_private_repo: bool = True
    report_to: str = "all"  # comet_ml
    tokenizer_max_length: int = 1024
    tokenizer_padding: 'bool|str' = True  # max length per batch
    use_gpu: bool = True
    optimizer: str = "adamw_torch"
    tokenizer_special_tokens: dict = None
    mixed_precision: str = None # "fp16"  #["no", "fp16", "bf16", "fp8"] or None
    # gradient_checkpointing: bool = False
    only_cpu: bool = False
    seed: int = 42
    run_id: str = None
    load_best_model: bool = True
    remove_cols_lst: list = None

    MAX_GPU_BATCH_SIZE: ClassVar = 8
    EVAL_BATCH_SIZE: ClassVar = 8

    def get_experiment(run_id):
        api_experiment = None
        experiment_id = None

        if run_id is not None:
            experiment_id = hashlib.sha1(run_id.encode("utf-8")).hexdigest()
            os.environ["COMET_EXPERIMENT_KEY"] = experiment_id

            api = comet_ml.API()  # Assumes API key is set in config/env
            api_experiment = api.get_experiment_by_key(experiment_id)

        if api_experiment is None:
            print("creating new experiment")
            return comet_ml.Experiment(project_name=os.getenv("COMET_PROJECT_NAME"), experiment_key=experiment_id)
        else:
            return comet_ml.ExistingExperiment(
                project_name=os.getenv("COMET_PROJECT_NAME"), log_env_details=True
            )
        
    # New Code #
    def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
        """Utility function for checkpointing model + optimizer dictionaries
        The main purpose for this is to be able to resume training from that instant again
        """
        checkpoint_state_dict = {
            "epoch": epoch,
            "last_global_step": last_global_step,
        }
        # Add extra kwargs too
        checkpoint_state_dict.update(kwargs)

        success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
        status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
        if success:
            logging.info(f"Success {status_msg}")
        else:
            logging.warning(f"Failure {status_msg}")
        return

    # New Code #
    def load_training_checkpoint(self, load_dir, tag=None, **kwargs):
        """Utility function for checkpointing model + optimizer dictionaries
        The main purpose for this is to be able to resume training from that instant again
        """
        _, checkpoint_state_dict = self.model.load_checkpoint(load_dir, tag=tag, **kwargs)
        epoch = checkpoint_state_dict["epoch"]
        last_global_step = checkpoint_state_dict["last_global_step"]
        del checkpoint_state_dict
        return (epoch, last_global_step)

    def __post_init__(self):
        self.experiment = TextClassificationWAccelerate.get_experiment(self.run_id)
        helpers.print_gpu_utilization()

        set_seed(self.seed)
        self.accelerator = Accelerator(cpu=self.only_cpu,
                                       mixed_precision=self.mixed_precision,
                                       log_with=self.report_to,
                                       project_dir=self.output_dir)

        self.logger = get_logger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            )
        self.logger.info(self.accelerator.state, main_process_only=False)
        
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.gradient_accumulation_steps = 1
        self.set_batch_size()
        self.set_dataset()

        self.tokenized_datasets = None
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = self.get_dataloaders()

        self.evaluator: EvaluationModule = evaluate.load(self.metric)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=len(self.id2label.keys()),
            id2label=self.id2label,
            label2id=self.label2id,
            return_dict=True)
        

    def set_dataset(self):  #ok
        print(self.dataset)
        if type(self.dataset) is str:
            self.dataset: DatasetDict = load_dataset(self.dataset)

        if self.remove_cols_lst is not None and len(self.remove_cols_lst) > 0:
            self.dataset = self.dataset.remove_columns(self.remove_cols_lst,)
        print(self.dataset)

        if self.undersample:
            self.accelerator.print("undersampling")
            self._random_undersample()

    def resample_ds(self, split):
        labels = np.array(self.dataset[split]['label'])

        print(np.unique(np.array(self.dataset[split]['label']),
                        return_counts=True))

        # Get the indices for each class
        label_indices: dict = {}
        min_length = -1
        for _label in np.unique(labels):
            label_indices[_label] = np.where(labels == _label)[0]
            length = len(label_indices[_label])
            if length < min_length or min_length == -1:
                min_length = length
                lowest_label = _label

        print("undersampling so all each class in the training set has length"
              f" {min_length}, same as class {lowest_label}")

        shuffled = {}
        for _label, _indices in label_indices.items():
            np.random.shuffle(_indices)
            shuffled[_label] = _indices[:min_length]

        balanced_indices = np.concatenate([v for _, v in shuffled.items()])
        self.dataset[split] = self.dataset[split].select(balanced_indices)

        print(np.unique(np.array(self.dataset[split]['label']),
                        return_counts=True))

    def _random_undersample(self):
        self.resample_ds(self.training_key)
        self.resample_ds(self.eval_key)
        self.resample_ds("test")
        return self.dataset

    def set_batch_size(self):
        self.batch_size = int(self.per_device_train_batch_size)

        # If the batch size is too big we use gradient accumulation
        if self.batch_size > TextClassificationWAccelerate.MAX_GPU_BATCH_SIZE and \
           self.accelerator.distributed_type != DistributedType.TPU:

            self.gradient_accumulation_steps = self.batch_size // TextClassificationWAccelerate.MAX_GPU_BATCH_SIZE
            self.batch_size = TextClassificationWAccelerate.MAX_GPU_BATCH_SIZE

        self.accelerator.print(f"gradient_accumulation_steps: {self.gradient_accumulation_steps}")

    def _tokenize_function(self, examples):
        # max_length=None => use the model max length (it's actually the default)
        if self.uncase:
            examples[self.text_col] = [text.lower() for text in examples[self.text_col]]

        outputs = self.tokenizer(examples[self.text_col],
                                 truncation=True,
                                 max_length=self.tokenizer_max_length,
                                 padding=self.tokenizer_padding)  # 'max_length', 'longest', True
        return outputs

    def _collate_fn(self, examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = 128 if self.accelerator.distributed_type == DistributedType.TPU else None
        # When using mixed precision we want round multiples of 8/16
        if self.accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif self.accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return self.tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def get_dataloaders(self):
        """
        Creates a set of `DataLoader`s for the `debateorg_{ideology}` dataset,
        using a predefined model as the tokenizer.
        """

        # Apply the method we just defined to all the examples in all the splits of the dataset
        # starting with the main process first:
        with self.accelerator.main_process_first():
            self.tokenized_datasets = self.dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=["text"]
            )

        self.tokenized_datasets = self.tokenized_datasets.rename_column("label",
                                                                        "labels")
        # self.tokenized_datasets = self.tokenized_datasets.remove_columns(self.tokenized_datasets["training"].column_names)
        # Instantiate dataloaders.
        train_dataloader: DataLoader = DataLoader(
            self.tokenized_datasets[self.training_key],
            shuffle=True,
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
            drop_last=True
        )

        eval_dataloader: DataLoader = DataLoader(
            self.tokenized_datasets[self.eval_key],
            shuffle=False,
            collate_fn=self._collate_fn,
            batch_size=TextClassificationWAccelerate.EVAL_BATCH_SIZE,
            drop_last=(self.accelerator.mixed_precision == "fp8"),
        )

        test_dataloader: DataLoader = DataLoader(
            self.tokenized_datasets["test"],
            shuffle=False,
            collate_fn=self._collate_fn,
            batch_size=TextClassificationWAccelerate.EVAL_BATCH_SIZE,
            drop_last=(self.accelerator.mixed_precision == "fp8"),
        )

        return train_dataloader, eval_dataloader, test_dataloader
    
    # New Code #
    def evaluate(self,  split_dataloader, extra_params):
        self.model.eval()
        for _, batch in enumerate(tqdm(split_dataloader)):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(self.accelerator.device)
            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = self.accelerator.gather_for_metrics(
                (predictions, batch["labels"])
                )
            self.evaluator.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = self.evaluator.compute(predictions=None,
                                             references=None,
                                             **extra_params)
        return eval_metric
    
    def train(self):
        try:
            # Initialize accelerator
            # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
            lr = self.learning_rate
            num_epochs = self.num_train_epochs
            extra_params = {"average": self.averaged_metric} if self.averaged_metric is not None else {}
            self.accelerator.print(f"extra_params: {extra_params}")
            
            # Instantiate the model (we build the model here so that the seed also control new weights initialization)

            # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
            # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
            # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
            self.model = self.model.to(self.accelerator.device)
            self.accelerator.print("*************** after setting model to accelerate **************")
            helpers.print_gpu_utilization()
            # Instantiate optimizer
            optimizer = AdamW(params=self.model.parameters(), lr=lr)

            # Instantiate scheduler
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=50,
                num_training_steps=(len(self.train_dataloader) * num_epochs) // self.gradient_accumulation_steps,
            )

            # Prepare everything
            # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
            # prepare method.

            self.model, optimizer, self.train_dataloader, self.eval_dataloader, lr_scheduler = self.accelerator.prepare(
                self.model,
                optimizer,
                self.train_dataloader,
                self.eval_dataloader,
                lr_scheduler
                )
            best_metric = None
            best_metric_checkpoint = None
            completed_steps = 0
            starting_epoch = 0

            # Now we train the model
            for epoch in range(num_epochs):
                helpers.print_gpu_utilization()

                self.accelerator.print(f"epoch {epoch} training...")
                self.model.train()
                for step, batch in enumerate(tqdm(self.train_dataloader)):
                    # We could avoid this line since we set the accelerator with `device_placement=True`.
                    batch.to(self.accelerator.device)
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss = loss / self.gradient_accumulation_steps
                    self.accelerator.backward(loss)
                    if step % self.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        completed_steps += 1

                self.accelerator.print(f"epoch {epoch} evaluating...")
                # here supports only 1 metric for now
                eval_metric = self.evaluate(self.eval_dataloader, extra_params)[self.metric]

                self.accelerator.print(f"epoch {epoch} - {self.metric}", eval_metric)
                
                try:
                    if self.experiment is not None:
                        self.experiment.log_metric(self.metric,
                                                eval_metric,
                                                epoch=epoch)
                except Exception as e:
                    self.accelerator.print("couldnt log metric", e)

                epoch_checkpoint_dir = f"epoch_{epoch}"
                if self.output_dir is not None:
                    epoch_checkpoint_dir = os.path.join(self.output_dir, epoch_checkpoint_dir)

                # New Code #
                # Tracks the best checkpoint and best metric
                if best_metric is None or best_metric < eval_metric:
                    best_metric = eval_metric
                    best_metric_checkpoint = epoch_checkpoint_dir
                    self.accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
                    self.accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")
                    self.accelerator.print("removing previously saved checkpoints...")
                    folders = glob(f"{self.output_dir}/*")
                    for f in folders:
                        if f != best_metric_checkpoint:
                            try:
                                shutil.rmtree(f)
                            except Exception as e:
                                print("error while trying to delete prev folders:", e)
                    
                    if self.save_strategy == "epoch":
                        self.accelerator.save_state(best_metric_checkpoint)
                if num_epochs == (epoch-1):
                    self.accelerator.print("saving the last state")
                    self.accelerator.save_state(epoch_checkpoint_dir)

            self.accelerator.print("Loading best model at "
                                   f"{best_metric_checkpoint}")
            self.accelerator.load_state(best_metric_checkpoint)
            re_eval = self.evaluate(self.eval_dataloader, extra_params)[self.metric]
            self.accelerator.print(f"evaluation on eval: {re_eval}")
            test_eval_val = self.evaluate(self.test_dataloader, extra_params)[self.metric]
            self.accelerator.print(f"evaluation on test set: {test_eval_val}")

        finally:
            self.accelerator.free_memory()
        # Loads the best checkpoint after the training is finished

    def _get_example(self, index):
        return self.tokenized_data[self.eval_key][index][self.text_col]

    # POST TRAINING
    def predict_all(self, text, task: str = "text-classification", use_current_model:bool=True):
        if use_current_model:
            result = self.model.predict(text) 
        else:
            classifier = pipeline(task, model=f"{self.model_org}/"
                                              f"{self.output_dir}")
            result = classifier(text)
        return result

    def predict_1label(self, text: str, use_current_model: bool = True):
        tokenizer = AutoTokenizer.from_pretrained(f"{self.model_org}"
                                                  f"/{self.output_dir}")
        inputs = tokenizer(text, return_tensors="pt")

        model = AutoModelForSequenceClassification.from_pretrained(f"{self.model_org}/{self.output_dir}")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        label = model.config.id2label[predicted_class_id]
        return label
    

    
    @staticmethod
    def compute_objective(predictions, targets):
        # Convert predictions and targets to numpy arrays if they are not already
        predictions = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
        targets = targets.numpy() if isinstance(targets, torch.Tensor) else targets
        
        # Calculate the macro F1 score
        macro_f1 = f1_score(targets, predictions, average='macro')
        
        return macro_f1