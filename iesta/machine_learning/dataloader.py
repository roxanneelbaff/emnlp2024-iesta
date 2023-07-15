import dataclasses
from typing import ClassVar, Dict

from sklearn.model_selection import train_test_split
import numpy
from tqdm import tqdm
import os

import iesta
import iesta.utils as utils
import iesta.processor as proc
import iesta.properties as properties
import pandas as pd
from iesta.machine_learning import dataloader
import cleantext
import iesta.properties as prop  
from tqdm import tqdm
from sklearn.utils.random import sample_without_replacement
import re

import pandas as pd
from ydata_profiling import ProfileReport
from ydata_profiling import ProfileReport
### HELPERS

def clean_argument(arg):
    arg = cleantext.clean(arg, 
            lower=False, 
            no_line_breaks=True, 
            no_urls=True, 
            no_emails=True, 
            no_phone_numbers=True, 
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            lang="en")
    # replace smileys
    # replace URLS
    return arg

def _apply_clean_txt(row, col):
    row["cleaned_text"] = clean_argument(row[col])
    return row

def apply_add_rounds(row, df__):
    row['rounds'] = df__["round"].tolist()
    return row


def get_previous_opposite_arg(df, debate_id, p_name, debate_round, text_col):
    prev_arg = ""
    if debate_round > 0:
        prev_arg_df = df[
            (df["debate_id"] == debate_id) & ~(df["p_name"] == p_name) & (df["round"] == (debate_round - 1))]
        if len(prev_arg_df) == 0:
            with open("../data/error_no_previous_args.txt", "a") as f:
                f.write(f"{debate_id},  p_name:{p_name}, round:{debate_round}\n")

        prev_arg = prev_arg_df[text_col].values[0] if len(prev_arg_df) > 0 else ""
    return prev_arg


def apply_add_prev_arg(row, original_df):
    debate_id = row['debate_id']
    p_name = row['p_name']
    debate_round = row['round']

    previous_opposite_arg = dataloader.get_previous_opposite_arg(original_df, debate_id, p_name, debate_round)
    row["current_arg_w_previous"] = f"{previous_opposite_arg} <MAIN-ARG> {row['argument']}"

    return row


def _get_rand_sample_indices(populaion,sample_num,random_state=42):
        return sample_without_replacement(populaion, sample_num, random_state=random_state)

def _get_sample_debates(df, split, sample_ratio:float = 0.3):
    debates = df[df['split'] == split]['debate_id'].unique().tolist()
    sample_size = round(len(debates) * sample_ratio)
    sample_indices = _get_rand_sample_indices(len(debates), sample_size)
    return [debates[x] for x in sample_indices]

## END OF HELPERS
@dataclasses.dataclass
class METHODOLOGY:
    EACH: ClassVar = "each" # Only this is used
    FIRST: ClassVar = "first"
    LAST: ClassVar = "last"
    CURRENT_PREVIOUS: ClassVar = "currentprevious"
    FIRST_LAST: ClassVar = "firstlast"

@dataclasses.dataclass
class LABELS:
    EFFECTIVE: ClassVar = "effective"
    INEFFECTIVE: ClassVar = "ineffective"
    PROVOCOTIVE: ClassVar = "provocative"
    OKAY: ClassVar = "okay"

    ALL: ClassVar = [EFFECTIVE, INEFFECTIVE, PROVOCOTIVE, OKAY]
    EFF_INEFF: ClassVar = [EFFECTIVE, INEFFECTIVE]

@dataclasses.dataclass
class IESTAData:
    ideology: str
    test_split: float = 0.3
    random_state: int = 2

    methodology: str = METHODOLOGY.EACH

    keep_labels: list = dataclasses.field(default_factory=lambda: LABELS.EFF_INEFF )

    ## not to use
    data_df: pd.DataFrame = None
    pivot_df: pd.DataFrame = None

    #evaluation_classfier_data_flag:int = dataloader.IESTAData._WITHOUT_EVAL_CLASSIFIER

    _ALL_DATA_: ClassVar = 0
    _ONLY_EVAL_CLASSIFIER_: ClassVar =1
    _WITHOUT_EVAL_CLASSIFIER: ClassVar = 2

    def _get_out_files_path(self):
        return os.path.join(properties.ROOT_PATH, "splitted",  f"methodology_{self.methodology}")

    
    def split_iesta_dataset_by_debate(self, force_reload: bool = False, profile:bool= False): 
        """
        parammeters:
            evaluation_classfier_data_flag: int 
                if 0 get all data
                if 1 get only 
        """ 

        def _apply_add_evaluation_classifier_data(row, debates):
            row['is_for_eval_classifier'] = True if row['debate_id'] in debates else False
            return row 
        
        def _apply_no_punc( row, col):
            row["text_no_punc"] = re.sub(r'[^\w\s]', '', row[col])
            row["text_no_punc_on_clean"] = re.sub(r'[^\w\s]', '', row["cleaned_text"])
            return row

        file_path = os.path.join(properties.ROOT_PATH,
                                 f"splitted_{self.ideology.lower()}"
                                 f"_debate_arguments"
                                 f"_effect_test{self.test_split}"
                                 f"_random{self.random_state}.parquet")
        print(file_path)
        tqdm.pandas()

        if os.path.isfile(file_path) and not force_reload:
            data_w_splits_df = pd.read_parquet(file_path)
            # GET data for the evaluation trainer for style
            print("The file for data_w_splits_df already exists")
        else: 
            processor = proc.Process()
            df, _ = processor.get_ideology_based_voter_participant_df(self.ideology)
            print(f"Original df len: {len(df)}")
            if self.keep_labels is not None and len(self.keep_labels)>0:
                df = df[df['effect'].isin(self.keep_labels)]
                print(f"After filtering effects {len(df)}")

            if 'cleaned_text' not in df.columns.tolist():
                print("Adding Cleaned text")
                df = df.apply(_apply_clean_txt, axis=1, args=("argument",))

            dissmiss_arr = []
            orig_dissmiss_arr = []
           
            with open(os.path.join(properties.ROOT_PATH, "dismiss_text.txt"), "r") as dismissedf:
                dissmiss_arr = list(pd.Series(dismissedf.read().splitlines()).str.lower())
                dissmiss_arr = list(set([re.sub(r'[^\w\s]', '', x) for x in dissmiss_arr]))
            
            with open(os.path.join(properties.ROOT_PATH, "dismiss_text.txt"), "r") as dismissedf:
                orig_dissmiss_arr = list(pd.Series(dismissedf.read().splitlines()).str.lower())

            df = df.apply(_apply_no_punc, axis=1, args=("argument",))
            df = df[~df["text_no_punc"].str.lower().isin(dissmiss_arr)]
            df = df[~df["text_no_punc_on_clean"].str.lower().isin(dissmiss_arr)]
            print(f"After filtering dismissed no_punc df len: {len(df)}")
            df = df[~df["argument"].str.lower().isin(orig_dissmiss_arr)]
            df = df[~df["cleaned_text"].str.lower().isin(orig_dissmiss_arr)]
            print(f"After filtering dismissed df len: {len(df)}")
            print(f"Profiling data")
            
            df["text_low"] = df["cleaned_text"].str.lower()
            if profile:
                profile = ProfileReport(df[["text_low"]], title="Profiling Report")
                profile.to_file( os.path.join(properties.ROOT_PATH, "profilers", f"{self.ideology}_all_data_low.html"))

                profile = ProfileReport(df[["cleaned_text"]], title="Profiling Report")
                profile.to_file( os.path.join(properties.ROOT_PATH, "profilers", f"{self.ideology}_all_data.html"))
                print(f"End of profiling")
    
            debates = df['debate_id'].unique()

            training_debate, testval_debates = train_test_split(debates, test_size=self.test_split,
                                                                random_state=self.random_state,
                                                                shuffle=True)
            validation_debates, test_debates = train_test_split(testval_debates, test_size=(1 / 3),
                                                                random_state=self.random_state,
                                                                shuffle=True)

            splits_df_lst = []
            splits_df_lst.append(pd.DataFrame({'debate_id': training_debate, 'split': ['training'] * len(training_debate)}))
            splits_df_lst.append(
                pd.DataFrame({'debate_id': validation_debates, 'split': ['validation'] * len(validation_debates)}))
            splits_df_lst.append(pd.DataFrame({'debate_id': test_debates, 'split': ['test'] * len(test_debates)}))

            debate_split_df = pd.concat(splits_df_lst)

            data_w_splits_df = df.merge(right=debate_split_df, how='left', on='debate_id', suffixes=('', 'extra'))

            # Validation
            assert len(debate_split_df) == len(debates)
            assert len(data_w_splits_df['debate_id'].unique()) == len(debates)
            for split in ["training", "validation", "test"]:
                assert len(data_w_splits_df[data_w_splits_df['split'] == split]['debate_id'].unique()) == \
                    len(debate_split_df[debate_split_df['split'] == split])
            ## Remove DISMISSED text

        if 'is_for_eval_classifier' not in data_w_splits_df.columns.tolist():
            print("is_for_eval_classifier is not in the columns, adding it - used to have data for style classification")

            training_debates_sample = _get_sample_debates(data_w_splits_df, 'training')
            validation_debates_sample = _get_sample_debates(data_w_splits_df, 'validation')
            test_debates_sample = _get_sample_debates(data_w_splits_df, 'test')
            sample_debates = training_debates_sample + validation_debates_sample + test_debates_sample
            data_w_splits_df = data_w_splits_df.apply(_apply_add_evaluation_classifier_data, axis=1, args=(sample_debates,))
        data_w_splits_df = data_w_splits_df[["id", "debate_id", "p_name",
                                             "effect", "category",
                                             "round", "argument",
                                             "cleaned_text",
                                             "is_for_eval_classifier", "split"]]
        data_w_splits_df.index.name = "idx"
        data_w_splits_df.to_parquet(file_path, index=True)

        for g, _df in data_w_splits_df.groupby(["is_for_eval_classifier"]):
            print(f"\n{g}")
            if profile:
                profile = ProfileReport(_df, title=f"PR {self.ideology} is_for_eval_classifier {g}")
                profile.to_file(os.path.join(properties.ROOT_PATH, "profilers", f"{self.ideology}_is-for-evaluator_{g}.html"))

            print(pd.crosstab(_df['split'], _df['effect']))

        split_effect_pivot_df = pd.crosstab(data_w_splits_df['split'],
                                            data_w_splits_df['effect'])
        print(f"All\n{split_effect_pivot_df}")
        
        return data_w_splits_df, split_effect_pivot_df

    def prepare_data_for_transformers(self, reload:bool = False):
        data_w_splits_df, split_effect_pivot_df = self.split_iesta_dataset_by_debate()

        result_df_lst = []
        
        path = self._get_out_files_path()

        tqdm.pandas()

        df_file = os.path.join(path, f"processed_data_{self.ideology.lower()}.parquet")

        if os.path.isfile(df_file) and not reload:
            iesta.logger.info("File already created. Loading file...")
            df = pd.read_parquet(df_file)
            split_effect_pivot_df = pd.crosstab(data_w_splits_df['split'], data_w_splits_df['effect'])
            return df, split_effect_pivot_df, df_file

        text_col_name = "cleaned_text"
        for effect_split, effect_split_df in data_w_splits_df.groupby(["effect", "split"]):
            # for an effect
            args_lst = []
            temp_df = pd.DataFrame()

            utils.create_folder(path)
            file = os.path.join(path, f"{self.ideology.lower()}_{effect_split[0]}_{effect_split[1]}.txt")

            if self.methodology == METHODOLOGY.EACH:
                args_lst = effect_split_df[text_col_name].tolist()
                df_ = effect_split_df.copy()
                result_df_lst.append(df_)

            elif self.methodology in [METHODOLOGY.FIRST, METHODOLOGY.LAST]:
                for group, df_ in effect_split_df.groupby(["debate_id", "p_name"]):
                    debate_round = df_["round"].min() if self.methodology == "first" else df_["round"].max()

                    df_ = df_[df_["round"] == debate_round]
                    assert len(df_[text_col_name].tolist()) == 1
                    result_df_lst.append(df_)
                    args_lst.extend(df_[text_col_name].tolist())

            elif self.methodology in [METHODOLOGY.FIRST_LAST]:
                for _, df_ in effect_split_df.groupby(["debate_id", "p_name"]):
                    df_ = df_[(df_["round"] == df_["round"].min()) | (df_["round"] == df_["round"].max())]
                    df_ = df_.sort_values(by="round")

                    arg = " <LAST> ".join(df_[text_col_name].tolist())
                    assert len(df_[text_col_name].tolist()) == 2 if (df_["round"].min() != df_["round"].max()) else \
                        len(df_[text_col_name].tolist()) == 1

                    temp_df = df_[-1:][['id',
                                        'debate_id',
                                        'p_name',
                                        'effect',
                                        'category',
                                        'round',
                                        'argument',
                                        'split']].copy()

                    temp_df = temp_df.apply(dataloader.apply_add_rounds, args=(df_,), axis=1, )
                    temp_df['argument'] = arg
                    result_df_lst.append(temp_df)
                    args_lst.append(arg)

            elif self.methodology in [METHODOLOGY.CURRENT_PREVIOUS]:
                # for group, df_ in data_w_splits_df.groupby(["debate_id", "round"]):
                df_ = effect_split_df.apply(dataloader.apply_add_prev_arg, args=(data_w_splits_df,), axis=1)
                args_lst.extend(df_["current_arg_w_previous"].tolist())
                text_col_name = "current_arg_w_previous"
                result_df_lst.append(df_)

            with open(file, 'w', encoding="utf8") as f:
                for arg in args_lst:
                    arg = f"<s>{arg}</s>"
                    f.write(f"{arg}\n")

        df = pd.concat(result_df_lst)
        df.index.name = "idx"
        # df= df.apply(dataloader._apply_clean_txt, args=(text_col_name,) , axis=1)
        df.to_parquet(df_file, index=True)
        return df, pd.crosstab(df['split'], df['effect']), df_file


    def get_training_data(self, reload=False):
        # Works only for EACH
        path = self._get_out_files_path()
        df_file = os.path.join(path,
                               f"{self.ideology.lower()}_training.parquet")

        if os.path.isfile(df_file) and not reload:
            iesta.logger.info("File already created. Loading file...")
            training_data = pd.read_parquet(df_file)
            return training_data, df_file
        print("getting training data only")
        data_w_splits_df, _ = self.split_iesta_dataset_by_debate()
        training_data = data_w_splits_df.copy()
        training_data = training_data[((training_data["is_for_eval_classifier"]==False)&(training_data["split"]=="training"))]

        utils.create_folder(path)
        training_data.to_parquet(df_file)
        return training_data, df_file


################### GENERIC FUNCTIONS ###################
from glob import glob
from iesta.machine_learning.feature_extraction import get_features_df


def load_training_data(ideology:str = "liberal", keep_labels=LABELS.EFF_INEFF, methodology:str=METHODOLOGY.EACH) -> Dict[str,pd.DataFrame]:
    dataloader = IESTAData(ideology=ideology,
                           keep_labels=keep_labels,
                           methodology=methodology)
    _, training_data_path = dataloader.get_training_data()
    training_data = pd.read_parquet(training_data_path)

    return training_data


def load_training_features_df(ideology: str = "liberal", keep_labels=LABELS.EFF_INEFF):#->Dict[str,pd.DataFrame] Dict[str, Dict[str, pd.DataFrame]]):
    path = "../data/extracted_features/"

    training_data = load_training_data(ideology=ideology, keep_labels=keep_labels, methodology=METHODOLOGY.EACH)

    feature_dfs = {}

    style_features_path = glob(f"{path}/{ideology}_style-features_5000/*.parquet")
    transformer_features_path = glob(f"{path}/{ideology}_transformer-features_100/*.parquet")

    empath_mpqa_df  = get_features_df(style_features_path, 5000, training_data)
    transformers_based_features_df = get_features_df(transformer_features_path, 100, training_data)

    difference = transformers_based_features_df.columns.difference(empath_mpqa_df.columns)
    feature_dfs =  empath_mpqa_df.merge(transformers_based_features_df[difference], right_index=True, left_index=True)
        
    return training_data, feature_dfs
