import dataclasses
from typing import ClassVar

from sklearn.model_selection import train_test_split
import numpy
from tqdm import tqdm
import os

import iesta
import iesta.utils as utils
import iesta.processor as proc
import iesta.properties as properties
import pandas as pd
import re
from iesta.machine_learning import dataloader
import cleantext

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


def get_previous_opposite_arg(df, debate_id, p_name, debate_round):
    prev_arg = ""
    if debate_round > 0:
        prev_arg_df = df[
            (df["debate_id"] == debate_id) & ~(df["p_name"] == p_name) & (df["round"] == (debate_round - 1))]
        if len(prev_arg_df) == 0:
            with open("../data/error_no_previous_args.txt", "a") as f:
                f.write(f"{debate_id},  p_name:{p_name}, round:{debate_round}\n")

        prev_arg = prev_arg_df["argument"].values[0] if len(prev_arg_df) > 0 else ""
    return prev_arg


def apply_add_prev_arg(row, original_df):
    debate_id = row['debate_id']
    p_name = row['p_name']
    debate_round = row['round']

    previous_opposite_arg = dataloader.get_previous_opposite_arg(original_df, debate_id, p_name, debate_round)
    row["current_arg_w_previous"] = f"{previous_opposite_arg} <MAIN-ARG> {row['argument']}"

    return row


def apply_binary_effect(row):
    row['binary_effect'] = "ineffective" if row['effect'] != "effective" else row['effect']

    return row


## END OF HELPERS
@dataclasses.dataclass
class METHODOLOGY:
    EACH: ClassVar = "each"
    FIRST: ClassVar = "first"
    LAST: ClassVar = "last"
    CURRENT_PREVIOUS: ClassVar = "currentprevious"
    FIRST_LAST: ClassVar = "firstlast"


@dataclasses.dataclass
class IESTAData:
    ideology: str
    test_split: float = 0.3
    abstract_effect: bool = False
    random_state: int = 2

    methodology: str = METHODOLOGY.EACH

    ## not to use
    data_df: pd.DataFrame = None
    pivot_df: pd.DataFrame = None
    pivot_binary_effect: pd.DataFrame = None


    def split_iesta_dataset_by_debate(self):
        def _abstract_effect(row):
            if row['effect'] != "effective":
                row['effect'] = "ineffective"
            return row

        abstract_st = "_abstracted" if self.abstract_effect else ""
        file_path = os.path.join(properties.ROOT_PATH,
                                 f"splitted_{self.ideology.lower()}"
                                 f"_debate_arguments{abstract_st}"
                                 f"_effect_test{self.test_split}"
                                 f"_random{self.random_state}.parquet")
        print(file_path)
        tqdm.pandas()

        if os.path.isfile(file_path):
            data_w_splits_df = pd.read_parquet(file_path)
            split_effect_pivot_df = pd.crosstab(data_w_splits_df['split'], data_w_splits_df['effect'])
            return data_w_splits_df, split_effect_pivot_df

        processor = proc.Process()
        df, _ = processor.get_ideology_based_voter_participant_df(self.ideology)
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

        # Abstract
        if self.abstract_effect:
            data_w_splits_df = data_w_splits_df.apply(_abstract_effect, axis=1)

        split_effect_pivot_df = pd.crosstab(data_w_splits_df['split'], data_w_splits_df['effect'])
        print(split_effect_pivot_df)
        data_w_splits_df.to_parquet(file_path)
        return data_w_splits_df, split_effect_pivot_df

    def prepare_data_for_transformers(self):
        data_w_splits_df, split_effect_pivot_df = self.split_iesta_dataset_by_debate()

        result_df_lst = []
        abstract_st = "abstracted" if self.abstract_effect else ""
        path = os.path.join(properties.ROOT_PATH, "splitted", abstract_st, f"methodology_{self.methodology}")

        tqdm.pandas()

        df_file = os.path.join(path, f"processed_data_{self.ideology.lower()}.parquet")

        if os.path.isfile(df_file):
            iesta.logger.info("File already created. Loading file...")
            df = pd.read_parquet(df_file)
            split_effect_pivot_df = pd.crosstab(data_w_splits_df['split'], data_w_splits_df['effect'])
            return df, split_effect_pivot_df, df_file

        text_col_name = "argument"
        for effect_split, effect_split_df in data_w_splits_df.groupby(["effect", "split"]):
            # for an effect
            args_lst = []
            temp_df = pd.DataFrame()

            utils.create_folder(path)
            file = os.path.join(path, f"{self.ideology.lower()}_{effect_split[0]}_{effect_split[1]}.txt")

            if self.methodology == METHODOLOGY.EACH:
                args_lst = effect_split_df["argument"].tolist()
                df_ = effect_split_df.copy()
                result_df_lst.append(df_)

            elif self.methodology in [METHODOLOGY.FIRST, METHODOLOGY.LAST]:
                for group, df_ in effect_split_df.groupby(["debate_id", "p_name"]):
                    debate_round = df_["round"].min() if self.methodology == "first" else df_["round"].max()

                    df_ = df_[df_["round"] == debate_round]
                    assert len(df_["argument"].tolist()) == 1
                    result_df_lst.append(df_)
                    args_lst.extend(df_["argument"].tolist())

            elif self.methodology in [METHODOLOGY.FIRST_LAST]:
                for group, df_ in effect_split_df.groupby(["debate_id", "p_name"]):
                    df_ = df_[(df_["round"] == df_["round"].min()) | (df_["round"] == df_["round"].max())]
                    df_ = df_.sort_values(by="round")

                    arg = " <LAST> ".join(df_["argument"].tolist())
                    assert len(df_["argument"].tolist()) == 2 if (df_["round"].min() != df_["round"].max()) else \
                        len(df_["argument"].tolist()) == 1

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
                    arg = f"<s>{dataloader.clean_argument(arg)}</s>"
                    f.write(f"{arg}\n")

        df = pd.concat(result_df_lst)
        df= df.apply(dataloader._apply_clean_txt, args=(text_col_name,) , axis=1)
        df.to_parquet(df_file)
        return df, pd.crosstab(df['split'], df['effect']), df_file

    def load(self, add_binary_effect:bool = True):
        self.data_df, self.pivot_df, file_path = self.prepare_data_for_transformers()
        
        if add_binary_effect:
            self.data_df = self.data_df.apply(dataloader.apply_binary_effect, axis=1)
            self.pivot_binary_effect = pd.crosstab(self.data_df['split'], self.data_df['binary_effect'])
        return self.data_df, self.pivot_df, file_path 

    def get_training_data(self, add_binary_effect:bool = True):
        _, _, _ = self.load(add_binary_effect=add_binary_effect)
        abstract_st = "abstracted" if self.abstract_effect else ""
        path = os.path.join(properties.ROOT_PATH, "splitted", abstract_st, f"methodology_{self.methodology}")
        df_file = os.path.join(path, f"processed_data_{self.ideology.lower()}_training.parquet")
        training_data = self.data_df[self.data_df["split"]== "training"].copy()
        utils.create_folder(path)
        training_data.to_parquet(df_file)
        return training_data, df_file


