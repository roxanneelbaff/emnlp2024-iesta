#!/home/elba_ro/repos/github/conf22-style-transfer/iesta_venv/bin/python3

from comet_ml import Experiment
import GPUtil
import torch

import os
import codecarbon
from iesta.machine_learning.dataloader import IESTAData, LABELS
from iesta.machine_learning.huggingface_loader import IESTAHuggingFace
from nlpaf.transformers.text_classification import TextClassification
from dotenv import load_dotenv, find_dotenv
import argparse
from huggingface_hub import login
from configs import all_configs

import random
import gc

global experiment
def init_comet(experiment_key=None, name=None):
    if len(experiment_key) < 32:
        random_str = ''.join(random.choice('0123456789ABCDEF') for i in range(32-len(experiment_key)))
        experiment_key = f"{experiment_key}R{random_str}"
    experiment_key = experiment_key[0:49] if len(experiment_key)>50 else experiment_key
    global experiment
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        experiment_key=experiment_key)
    experiment.set_name(name)
    return experiment
    

def reset():
    #gc.collect() 
    torch.cuda.empty_cache()
    torch.backends.cuda.max_split_size_mb = 128
    print(GPUtil.showUtilization())
    found = load_dotenv(find_dotenv())
    print(f"dotenv was {found}")


def check_cuda():
    use_cuda = torch.cuda.is_available()
    #CUDA_VISIBLE_DEVICES

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
    else:
        print('no cuda')


def run_experiment(config_key: str):
    global experiment
    try:
        reset()
  
        login(os.getenv('HUGGINGFACE_TOKEN'), add_to_git_credential=True)
        print(f"Running Experiments for key {config_key}")
        config_dict = all_configs[config_key]
        print(config_dict["output_dir"].split("/")[-1].replace("_", ""))
        experiment = init_comet(config_dict["output_dir"].split("/")[-1].replace("_", ""), config_dict["output_dir"])
        
        check_cuda()
        experiment.log_parameters(config_dict)
        data_object = IESTAData(ideology=config_dict["ideology"], keep_labels = LABELS.EFF_INEFF)
        huggingface_dataset = IESTAHuggingFace(data_object)

        dataset_name = huggingface_dataset.get_dataset_name(is_for_style_classifier=config_dict["is_for_style_classifier"])
        print(dataset_name)
        trainer = TextClassification(
            dataset_name,
            id2label=  IESTAHuggingFace._ID2LABEL_,
            label2id= IESTAHuggingFace._LABEL2ID_,
            training_key="training",
            eval_key="validation",
            text_col="text",
            uncase=config_dict["uncase"],
            undersample=config_dict["undersample"],
            pretrained_model_name=config_dict["pretrained_model_name"],#"microsoft/deberta-base",
            metric="f1",
            averaged_metric="macro",
            model_org="notaphoenix",  # only if you want to upload your model to hf
            output_dir=config_dict["output_dir"],
            learning_rate= config_dict["learning_rate"],#5e-6,
            per_device_train_batch_size=config_dict["per_device_train_batch_size"],
            per_device_eval_batch_size= config_dict["per_device_eval_batch_size"],
            num_train_epochs= config_dict["num_train_epochs"],
            weight_decay= config_dict["weight_decay"],
            evaluation_strategy= config_dict["evaluation_strategy"],
            save_strategy= config_dict["save_strategy"],
            load_best_model_at_end= True,
            push_to_hub= config_dict["push_to_hub"],
            hub_private_repo= True,
            report_to="comet_ml"  # comet_ml
            )
        

        if config_dict["search_hp"]:
            hpsearch_bestrun = trainer.search_hyperparameters( n_trials= 5,
                                        run_on_subset= False,
                                        direction="maximize",
                                        load_best_params=True,
                                        hp_space_func=config_dict["optuna_hp_func"]
                                        )
            print(f"********** BEST PARAMS for experiment {config_key} **********")
            for n, v in hpsearch_bestrun.hyperparameters.items():
                print(f" {n}:{v}")
        else:
            trainer.train()
            score = trainer.evaluate()
            experiment.log_metric("test_score", score)
    
    finally:
        experiment.end()


def main():
   # tracker = codecarbon.EmissionsTracker(log_level="error", tracking_mode="process", output_dir="codecarbon")

    try:
        #tracker.start()
        parser = argparse.ArgumentParser()
        parser.add_argument("-k", "--experimentkey", type=str)
        args = parser.parse_args()

        run_experiment(args.experimentkey)
    finally:
        #tracker.stop()
        global experiment
        experiment.end()
        #print(f"FINAL EMMISION {tracker.final_emissions}")

if __name__ == "__main__":
    main()