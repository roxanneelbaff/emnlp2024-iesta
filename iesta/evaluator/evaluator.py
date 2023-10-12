import dataclasses
import textwrap
import numpy as np
import pandas as pd
from transformers import pipeline

from nltk.tokenize import word_tokenize
from collections import Counter

import pandas as pd


from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
from langdetect import detect
from pathlib import Path

from datasets.combine import concatenate_datasets

from iesta.evaluator.generation_processor import process_llm_generated_args
from scipy.stats import pointbiserialr


@dataclasses.dataclass
class Evaluator:
    model_type: str  # either chatgpt or llamav2
    ideology: str
    shot_num: int
    steered: str = ""  # _steered_mean0.2

    ## exceptional - LIWC features_path
    feature_liwc_path: str = None
    root_path: str = "../data"
    generated_args_path = "llms_out/new"

    def __post_init__(self):
        self.key = (
            f"{self.ideoloy}_{self.model_type}_{self.shot_num}shot{steered}"
        )
        self.filename: str = (
            f"{self.root_path}/{self.generated_args_path}/{self.key}.jsonl"
        )

        print("postprocessing the generated arguments...")
        self.data = process_llm_generated_args(self.filename)

        print("fetching original ineffective arguments...")
        self.original_data_df = self.get_test_data()
        print("Calculating the toxicity scores for Ineffective arguments...")
        if "toxic" not in self.original_data_df.columns.tolist():
            toxicity_scores = self.score_toxicity(
                self.original_data_df["text"].values.tolist()
            )
            toxicity_scores_df = pd.DataFrame(toxicity_scores)
            self.original_data_df["toxic"] = toxicity_scores_df["toxic"]
        self.merged_df = pd.merge(
            self.data,
            self.original_data_df[self.ideology][
                [
                    "category",
                    "round",
                    "debate_id",
                    "idx",
                    "toxic",
                ]
            ],
            how="inner",
            on="idx",
        )

        def _add_toxic_lbl(row):
            row["toxic_str"] = "Toxic" if row["toxic"] >= 0.5 else "Not Toxic"
            row["success_int"] = 1 if row["success"] else 0
            return row

        self.merged_df = self.merged_df.apply(
            _add_toxic_lbl,
            axis=1,
        )
        print("ANALYSIS I - Failed to respond")
        print(
            "ANALYSIS I.a - Descriptive statistics"
            " Top 10-grams for failed arg top_ngrams_count_failed_response"
            "count_success_per_prompt_df\n"
            "count_success_per_category_df\n"
            "count_success_per_category_df\n"
            "count_success_per_toxic_df\n"
        )

        self.count_success_per_prompt_df = self.get_cross_counts(
            idx="prompt", col="success"
        )
        self.count_success_per_category_df = self.get_cross_counts(
            idx="category", col="success"
        )
        self.count_type_per_prompt_df = self.get_cross_counts(
            idx="prompt", col="type"
        )

        self.count_success_per_toxic_df = self.get_cross_counts(
            idx="toxic_str", col="success"
        )

        self.top_ngrams_count_failed_response = (
            self.analyze_failed_output_w_ngrams(n=10, top=30)
        )
        self.failed_ratio = (
            self.data[~self.data["success"]] * 100 / float(len(self.data))
        )
        self.total = len(self.data)
        self.failed_num = len(self.data[~self.data["success"]])

        self.corr_toxicity_no_response = self.calc_corr_toxicity_no_response()

        # Failed per prompt

    def get_test_data(
        self,
        effect="ineffective",
    ):
        out_path = f"{self.root}/test_{self.ideology}_{effect}.csv"
        if Path(out_path).is_file():
            print("original file found")
            return pd.read_csv(out_path)
        print("original file NOT found")
        label2id_dict = {
            "ineffective": 0,
            "effective": 1,
        }
        seed = 2062021
        name: str = f"notaphoenix/debateorg_w_effect_for_{self.ideology}"
        dataset: Dataset = load_dataset(name, split="test")
        dataset = dataset.filter(
            lambda x: x["label"] == label2id_dict[effect]
        ).shuffle(seed=seed)

        if len(dataset) > 500:
            dataset = dataset.select(range(500))

        print(f"{len(dataset)} before len filter")
        dataset = dataset.filter(
            lambda x: len(x["text"].split(" ")) > 10
            and len(x["text"].split(" ")) <= 1024
            and x["idx"] != 64707
            and detect(x["text"]) == "en"
        )
        print(f"{len(dataset)} after len filter")

        while len(dataset) < 500:
            idxes = dataset.to_pandas()["idx"].values.tolist()
            dataset_extra: Dataset = load_dataset(name, split="test")
            dataset_extra = dataset_extra.filter(
                lambda x: len(x["text"].split(" ")) > 10
                and len(x["text"].split(" ")) <= 1024
                and x["idx"] != 64707
                and detect(x["text"]) == "en"
            )

            dataset_extra = dataset_extra.filter(
                lambda x: x["label"] == label2id_dict[effect]
                and ["idx"] not in idxes
            ).shuffle(seed=seed)
            dataset_extra = dataset_extra.select(range(500 - len(dataset)))
            print(f"{len(dataset_extra)} of extra")
            dataset = concatenate_datasets([dataset, dataset_extra])

        dataset.to_pandas().to_csv(out_path)
        return dataset.to_pandas()

    def calc_corr_toxicity_no_response(
        self, corr_with_arr: list = ["prompt", "category"]
    ):
        result = []

        def calc_corr(df_):
            success_arr = [
                (1 if x else 0) for x in df_["success"].values.tolist()
            ]
            binary_categorical_data = np.array(success_arr)
            numeric_data = np.array(df_["toxic"].values.tolist())

            corr, p_value = pointbiserialr(
                numeric_data, binary_categorical_data
            )
            return corr, p_value

        for group in corr_with_arr:
            for g, experiment in self.merged_df.groupby(group):
                corr, p_value = calc_corr(experiment)
                result.append(
                    {
                        "experiment": self.key,
                        "group_value": g,
                        "group_type": group,
                        "correlation": corr,
                        "p-value": p_value,
                        "<0.05": p_value < 0.05,
                        "<0.00001": p_value < 0.00001,
                    }
                )

        corr, p_value = calc_corr(self.merged_df)
        result.append(
            {
                "experiment": self.key,
                "group_value": "",
                "group_type": "",
                "correlation": corr,
                "p-value": p_value,
                "<0.05": p_value < 0.05,
                "<0.00001": p_value < 0.00001,
            }
        )
        return pd.DataFrame(result)

    def get_cross_counts(self, idx="prompt", col="success"):
        col_failure_dict = {"success": False, "type": "refused to respond"}
        df = pd.crosstab(
            self.merged_df[idx], self.merged_df[col], margins=True
        )
        if col in col_failure_dict.keys():
            df["fail_perc"] = round(
                df[col_failure_dict[col]] * 100 / df["All"], 2
            )
        return df

    def classify_effectiveness(self):
        #  Load model from hf and classify
        pass

    def extract_style_features(self):
        # 1. extract stykle features
        # 2. save significance
        pass

    def score_style():
        pass

    def score_morality():
        # like malitiousness
        # Expected: LLM significantly decrease it
        pass

    def score_stance():
        # percent of correct ones
        pass

    def score_toxicity(texts: list[str], truncate_txt=False):
        toxigen_roberta = pipeline(
            "text-classification",
            model="tomh/toxigen_roberta",
            top_k=None,
            truncation=True,
            max_length=512,
        )

        def reformat(
            text_scores: list[dict],
        ):  # [{'label': 'LABEL_0', 'score': 0.5929976105690002}, {'label': 'LABEL_1', 'score': 0.40700244903564453}]
            map_ = {"LABEL_0": "non_toxic", "LABEL_1": "toxic"}
            return {map_[x["label"]]: x["score"] for x in text_scores}

        if not truncate_txt:
            toxic_scores = []

            for txt in texts:
                chunks = textwrap.wrap(
                    txt,
                    width=512,
                    placeholder="",
                )
                chunk_res = toxigen_roberta(chunks)
                chunk_res_formatted = pd.DataFrame(
                    [reformat(x) for x in chunk_res]
                )

                highest_toxic_score = chunk_res_formatted.loc[
                    chunk_res_formatted["toxic"].idxmax()
                ].to_dict()
                toxic_scores.append(highest_toxic_score)
        else:
            toxic_scores = toxigen_roberta(texts)
            toxic_scores = [reformat(x) for x in toxic_scores]

        return pd.DataFrame(toxic_scores)["toxic"]

    def analyze_failed_output_w_ngrams(self, n=10, top: int = 30):
        # Tokenize the text data
        text_data = self.data[~self.data["success"]]["generated"]
        tokens = []
        for sentence in text_data:
            sentence = re.sub("[^a-zA-Z0-9]+", " ", sentence).lower()
            sentence = sentence.replace("and", "")
            sentence = sentence.replace("that", "")
            sentence = sentence.replace("to", "")

            tokens += word_tokenize(sentence)

        # Generate all possible 3-grams
        n_grams = []
        for i in range(len(tokens) - (n - 1)):
            n_grams.append(tuple(tokens[i : i + n]))

        # Count the frequency of each 3-gram
        frequencies = Counter(n_grams)

        # Sort the 3-grams by frequency
        top_n_grams = sorted(frequencies.items(), key=lambda x: x[1])[-top:]

        # Print or store the results
        return top_n_grams
