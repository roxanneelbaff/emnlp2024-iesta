

from typing import ClassVar
from datasets import load_dataset
from tqdm import tqdm
from datasets.dataset_dict import Dataset, DatasetDict
from iesta.machine_learning.dataloader import IESTAData
import dataclasses
import os


@dataclasses.dataclass
class IESTAHuggingFace():
    iesta_dataset: IESTAData
    _DATASET_NAME_DIC_: ClassVar = {
        "LABELLED": "debateorg_w_effect_for_{ideology}",
        "LABELLED_FOR_STYLE_CLASSIFIER": "debateorg_w_effect_for_{ideology}_subset",
        "PER_EFFECT": "debateorg_{effect}_args_for_{ideology}"
    }
    def upload_w_labels(self, is_for_style_classifier:bool,  effect_label:str = "effect", text_col:str = "cleaned_text", prefix:str = "_effective_"):
        _KEY_  = "LABELLED_FOR_STYLE_CLASSIFIER" if is_for_style_classifier else "LABELLED"
        dataset_name= f"notaphoenix/{IESTAHuggingFace._DATASET_NAME_DIC_[_KEY_].format(ideology=self.iesta_dataset.ideology)}"

        try:
            labelled_dataset = load_dataset(dataset_name, use_auth_token=True) 
        except FileNotFoundError as e:
            print("dataset was not found on hugging face.")
            print("creating huggingface dataset from local dataset")

            data_w_splits_df, _ = self.iesta_dataset.split_iesta_dataset_by_debate()
            _df  = data_w_splits_df.copy()

            if is_for_style_classifier:
                _df = _df[_df['is_for_eval_classifier'] == True].copy()
            else:
                _df = _df[_df['is_for_eval_classifier'] == False].copy()

            _df = _df.rename({effect_label:'label', text_col:'text'},axis='columns')
            print(f"The columns are {_df.columns.tolist()}")

            data_splits: dict ={}
            for split,split_df in _df.groupby('split'):
                data_splits[split] = Dataset.from_pandas(split_df[["text", "label"]])
                
            labelled_dataset: DatasetDict = DatasetDict(data_splits)
            

            labelled_dataset.push_to_hub(dataset_name, private=True )
            labelled_dataset =  load_dataset(dataset_name, use_auth_token=True) 
        return labelled_dataset

    def upload_by_effect(self, effect, effect_label:str = "binary_effect", prefix:str = "_"):
        dataset_name= f"notaphoenix/iesta{prefix}{self.iesta_dataset.ideology}_{effect}"

        try:
            self.dataset = load_dataset(dataset_name, use_auth_token=True) 
        except FileNotFoundError as e:
            print("dataset was not found on hugging face.")
            print("creating huggingface dataset from local dataset")
            path = self.iesta_dataset._get_out_files_path()
            data_w_splits_df, _ = self.iesta_dataset.split_iesta_dataset_by_debate()
            _df  = data_w_splits_df[data_w_splits_df[effect_label] == effect].copy()

            data_files: dict ={}
            for s in _df['split'].unique():
                data_files[s] = os.path.join(path, f"{self.iesta_dataset.ideology.lower()}_{effect}_{s}.txt")
                
            ds: DatasetDict = load_dataset("text", data_files=data_files)
            parquet_data_files: dict = {}
            for split, dataset in ds.items():
                fname:str = f"../data/temp_hf/{split}_{self.iesta_dataset.ideology}_{effect}.parquet"
                dataset.to_parquet(fname)
                parquet_data_files[split]= fname

            self.hf_dataset = load_dataset("parquet", data_files=parquet_data_files)

            self.hf_dataset.push_to_hub(dataset_name, private=True )

    def get_hf_sample(self, split:str="train", count:int=5):
        samples = self.iesta_dataset.dataset[split].shuffle().select(range(count))
        return samples