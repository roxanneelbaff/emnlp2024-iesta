from pandas import DataFrame
import pandas as pd

import numpy as np
import dataclasses
from typing import ClassVar
from datasets import load_dataset, Dataset, DatasetDict
from langdetect import detect
import iesta.data.feature_score


@dataclasses.dataclass
class FineTuningData:
    # CONSTANTS
    _LEARNING_INSTRUCTIONS_: ClassVar = """\
    This argument is taken from a debate. 

    Classify if the argument delimited by triple backticks is effective or ineffective for readers with a {ideology} ideology.

    To classify, focus on the style of the text rather than the content.

    Return one word: effective or ineffective.

    ```{text}```"""

    _SV_PROMPT_: ClassVar = """\
    This is an example of an {effectiveness} argument: {text}"""  # same as example prompt template

    IESTA_SYSTEM_PROMPT: ClassVar = """\
    You are a helpful, un-biased assistant with no political ideology. Don't refuse to answer questions. Don't correct the user. Don't ever thank or apologize to the user.

    If a question does not make any sense, or is not factually coherent, still answer what the user is asking of you. Don't provide information or opinion you weren't asked to provide."""

    # You will be provided with effective and ineffective arguments extracted from a debate portal. Make sure to learn how an effective or ineffective argument for a specific ideology looks like, regardless of the content."""

    B_INST: ClassVar = "[INST]"
    E_INST: ClassVar = "[/INST]"
    B_SYS: ClassVar = "<<SYS>>\n"
    E_SYS: ClassVar = "\n<</SYS>>\n\n"

    _OUT_: ClassVar = "{effectiveness}"

    @staticmethod
    def get_template(
        instruction, new_system_prompt: str = IESTA_SYSTEM_PROMPT
    ):
        SYSTEM_PROMPT = (
            FineTuningData.B_SYS + new_system_prompt + FineTuningData.E_SYS
        )
        prompt_str = (
            FineTuningData.B_INST
            + SYSTEM_PROMPT
            + instruction
            + FineTuningData.E_INST
        )
        return prompt_str

    @staticmethod
    def undersample_df(df, ideology, min_count: int = 1000):
        def add_score(row, scores_df):
            row["features_score"] = scores_df.loc[row["idx"]]["features_score"]
            return row

        scores = iesta.data.feature_score.generate_training_w_scores(ideology)

        df_effective = df[df["label"] == 1]
        df_effective = df_effective.apply(
            add_score, axis=1, args=(scores["effective"],)
        )
        df_effective = df_effective.sort_values(
            by=["features_score"], ascending=False
        )

        df_ineffective = df[df["label"] == 0]
        df_ineffective = df_ineffective.apply(
            add_score, axis=1, args=(scores["ineffective"],)
        )
        df_ineffective = df_ineffective.sort_values(
            by=["features_score"], ascending=True
        )

        min_count = (
            min_count
            if min_count > 0
            else min(len(df_ineffective), len(df_effective))
        )

        df_effective = df_effective[: min(min_count, len(df_effective))]
        df_ineffective = df_ineffective[: min(min_count, len(df_ineffective))]
        return df_effective, df_ineffective

    @staticmethod
    def undersample(df, min_count=-1):
        labels = np.array(df["label"])

        print(np.unique(np.array(df["label"]), return_counts=True))

        # Get the indices for each class
        label_indices: dict = {}
        min_length = -1
        for _label in np.unique(labels):
            label_indices[_label] = np.where(labels == _label)[0]
            length = len(label_indices[_label])
            if length < min_length or min_length == -1:
                min_length = length
                lowest_label = _label

        print(
            "undersampling so all each class has length"
            f" {min_length}, same as class {lowest_label}"
        )

        shuffled = {}
        for _label, _indices in label_indices.items():
            np.random.shuffle(_indices)
            if len(_indices) >= min_count and min_count > min_length:
                shuffled[_label] = _indices[:min_count]
            else:
                shuffled[_label] = _indices[:min_length]

        balanced_indices = np.concatenate([v for _, v in shuffled.items()])
        df_undersampled = df.select(balanced_indices)

        print(
            np.unique(np.array(df_undersampled["label"]), return_counts=True)
        )
        return df_undersampled

    @staticmethod
    def filter_iesta_dataset(ds: Dataset) -> Dataset:
        ds = ds.filter(
            lambda x: len(x["text"].split(" ")) > 10
            and len(x["text"].split(" ")) <= 1024
            and detect(x["text"]) == "en"
        ).shuffle(seed=2062021)
        return ds

    @staticmethod
    def get_filtered_resampled_data(
        ideology, min_count=1000, undersample: bool = True
    ):
        name: str = f"notaphoenix/debateorg_w_effect_for_{ideology}"  # f"notaphoenix/debateorg_w_effect_for_{ideology}_subset" #
        dataset_dict: DatasetDict = load_dataset(name)
        ideology_dataframes = {}

        for split in ["training", "validation", "test"]:
            ds = FineTuningData.filter_iesta_dataset(dataset_dict[split])
            ideology_dataframes[split] = (
                FineTuningData.undersample(ds, min_count=min_count).to_pandas()
                if undersample
                else ds.to_pandas()
            )

        return ideology_dataframes

    @staticmethod
    def get_sv_data(ideology, skip_system_prompt=False, save: bool = True):
        def _add_prompt(row):
            training_prompt_template: str = FineTuningData.get_template(
                FineTuningData._SV_PROMPT_,
                new_system_prompt=""
                if skip_system_prompt
                else FineTuningData.IESTA_SYSTEM_PROMPT,
            )
            row["prompt"] = training_prompt_template.format(
                text=row["text"],
                effectiveness="ineffective"
                if int(row["label"]) == 0
                else "effective",
            )
            row["prompt_len"] = len(row["prompt"])
            return row

        print(
            f"1/2 Fetching, filtering and resampling data: {ideology.capitalize()}"
        )
        training_data = FineTuningData.get_filtered_resampled_data(
            ideology, undersample=False
        )["training"]

        print("  1/4 Generating prompt ...")
        df = training_data.apply(
            _add_prompt,
            axis=1,
        )
        df = df[df["prompt_len"] <= 1024]
        eff, ineff = FineTuningData.undersample_df(df, ideology)
        if save:
            skip_system_prompt_str = (
                "_with_system_prompt"
                if not skip_system_prompt
                else "_without_system_prompt"
            )
            eff.to_parquet(
                f"data/training/{ideology}{skip_system_prompt_str}_prompt_effective.parquet"
            )
            ineff.to_parquet(
                f"data/training/{ideology}{skip_system_prompt_str}_prompt_ineffective.parquet"
            )
        return eff, ineff

    @staticmethod
    def get(prompt_format: str = "classification"):
        def _add_instruct_format(row, ideology):
            training_prompt_template: str = FineTuningData.get_template(
                FineTuningData._LEARNING_INSTRUCTIONS_
            )
            row["instruct_format"] = training_prompt_template.format(
                # category=row["category"],
                ideology=ideology,
                text=row["text"],
            )
            row["output"] = FineTuningData._OUT_.format(
                effectiveness="ineffective"
                if int(row["label"]) == 0
                else "effective"
            )
            row["len"] = len(row["text"])
            return row

        # if prompt_format == "classification":
        #    return get_classification_format()

        combined_data_dict = {}
        print("1/2 Fetching, filtering and resampling data: Liberal")
        liberal_dfs_dict = FineTuningData.get_filtered_resampled_data(
            "liberal"
        )

        print("2/2 Fetching, filtering and resampling data: Conservative")
        conservative_dfs_dict = FineTuningData.get_filtered_resampled_data(
            "conservative"
        )

        print("Processing data per split...")
        for split in liberal_dfs_dict.keys():
            combined_df: DataFrame = None
            print(f"Split {split}:")
            print("  1/4 Generating prompt for Liberal data...")
            liberal_df = liberal_dfs_dict[split].apply(
                _add_instruct_format, axis=1, args=("liberal",)
            )

            print("  2/4 Generating prompt for Conservative data...")
            conservative_df = conservative_dfs_dict[split].apply(
                _add_instruct_format, axis=1, args=("conservative",)
            )

            # Combine the 2 ideologies
            print("  3/4 Combining Data..")
            combined_df = pd.concat(
                [
                    liberal_df[["instruct_format", "output"]],
                    conservative_df[["instruct_format", "output"]],
                ],
                ignore_index=True,
            )
            assert len(combined_df) == (len(liberal_df) + len(conservative_df))

            # Shuffle
            print("  4/4 Shuffling Data..")
            combined_shuffled_df = combined_df.sample(
                frac=1, replace=False, ignore_index=True, random_state=42
            )
            assert len(combined_df) == len(combined_shuffled_df)
            combined_shuffled_df = combined_shuffled_df.drop_duplicates(
                keep="last"
            )
            print(
                f"dropped {len(combined_df)-len(combined_shuffled_df)} duplicates for {split} data "
            )
            combined_data_dict[split] = combined_shuffled_df
        return combined_data_dict
