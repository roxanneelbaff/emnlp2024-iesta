import dataclasses
import re
import textwrap
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from transformers import pipeline

from nltk.tokenize import word_tokenize
from collections import Counter

import pandas as pd


from datasets import load_dataset, Dataset
from tqdm import tqdm
from langdetect import detect
from pathlib import Path

from datasets.combine import concatenate_datasets
from iesta import utils
from iesta.data.feature_extraction import IESTAFeatureExtractionPipeline
from iesta.data.feature_score import get_top_features
from iesta.data.huggingface_loader import IESTAHuggingFace

from iesta.evaluator.generation_processor import process_llm_generated_args
from scipy.stats import pointbiserialr


@dataclasses.dataclass
class Evaluator:
    model_type: str  # either chatgpt or llamav2
    ideology: str
    shot_num: int
    steered: str = ""  # _steered_mean0.2

    # exceptional - LIWC features_path
    feature_liwc_path: str = None
    root_path: str = ""  # or ../
    generated_args_path = "data/llms_out/new"

    def __post_init__(self):
        self.key = (
            f"{self.ideology}_{self.model_type}_{self.shot_num}shot{self.steered}"
        )
        self.filename: str = (
            f"{self.root_path}{self.generated_args_path}/{self.key}.jsonl"
        )

        print(f"postprocessing the generated arguments {self.filename}...")
        self.data = process_llm_generated_args(self.filename,
                                               root=self.root_path)

        print("fetching original ineffective arguments...")
        self.original_data_df = self.get_test_data()
        print(f"Calculating the toxicity scores for Ineffective arguments - existing cols {self.original_data_df.columns.tolist()}...")
        print(f"{type(self.original_data_df['text'].values.tolist())}")
        
        if "toxic" not in self.original_data_df.columns.tolist():
            print("Toxicity from scratch")
            toxicity_scores = self.score_toxicity(
                self.original_data_df["text"].values.tolist()
            )
            toxicity_scores_df = pd.DataFrame(toxicity_scores)
            self.original_data_df["toxic"] = toxicity_scores_df["toxic"]

            out_path = f"{self.root_path}data/test_{self.ideology}_ineffective.csv"
            self.original_data_df.to_csv(out_path, index=False)
        else: 
            print("Toxicity score already calculated")    
        print("Toxicity score just calculated")

        print("Merging original and generated data...")
        self.merged_df = pd.merge(
            self.original_data_df[[
                    "category",
                    "round",
                    "debate_id",
                    "idx",
                    "toxic",
                ]],
            self.data,
            
            how="inner",
            on="idx",
        )
        print(f"MERGED DATA SHOULD BE 500 {len(self.merged_df)}")

        def _add_toxic_lbl(row):
            row["toxic_str"] = "Toxic" if row["toxic"] >= 0.5 else "Not Toxic"
            row["success_int"] = 1 if row["success"] else 0
            return row

        self.merged_df = self.merged_df.apply(
            _add_toxic_lbl,
            axis=1,
        )

        def add_has_ideology_prompt(row):
            row["has_ideology_prompt"] = row["prompt"].find("ideology") >-1
            return row
        self.merged_df = self.merged_df.apply(add_has_ideology_prompt, axis=1)

        print("ANALYSIS I - Failed to respond")
        print(
            "ANALYSIS I.a - Descriptive statistics"
            
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

        self.count_success_per_ideology_prompt_df = self.get_cross_counts(
            idx="has_ideology_prompt", col="success"
        )

        self.count_success_per_toxic_df = self.get_cross_counts(
            idx="toxic_str", col="success"
        )

        self.top_ngrams_count_failed_response = (
            self.analyze_failed_output_w_ngrams(n=10, top=30)
        )
        self.failed_ratio = (
            len(self.data[~self.data["success"]]) * 100 / float(len(self.data))
        )
        self.total = len(self.data)
        self.failed_num = len(self.data[~self.data["success"]])

        self.corr_toxicity_no_response = self.calc_corr_toxicity_no_response()


        # Failed per prompt

    def get_test_data(
        self,
        effect="ineffective",
    ):
        print("Getting Original Data...")
        out_path = f"{self.root_path}data/test_{self.ideology}_{effect}.csv"
        if Path(out_path).is_file():
            print("original file found")
            return pd.read_csv(out_path)
        print(f"original file NOT found {out_path}")

        indices_path = f"{self.root_path}data/out/{self.ideology}_idx.csv"

        preset_indices = []
        print(f"{indices_path}")
        assert Path(indices_path).is_file()
        print("original INDICESS found")
        preset_indices = pd.read_csv(indices_path)["idx"].values.tolist()[:500]
        
        seed = 2062021
        name: str = f"notaphoenix/debateorg_w_effect_for_{self.ideology}"
        dataset: Dataset = load_dataset(name, split="test")

        dataset = dataset.filter(
            lambda x: x["label"] == IESTAHuggingFace._LABEL2ID_[effect]
        ).shuffle(seed=seed)

        assert len(preset_indices) > 0
        print(f"filtering using Indices with len {len(preset_indices)}")
        dataset = dataset.filter(
            lambda x: x["idx"] in preset_indices
            )
        print(f"using preset indices {len(dataset)}")
        df = dataset.to_pandas()
        df.to_csv(out_path, index=False)
        assert len(df) == 500
        return df

    def calc_corr_toxicity_no_response(
        self, corr_with_arr: list = ["has_ideology_prompt", "category"]
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
            ) if col_failure_dict[col] in df.columns.to_list() else 0.0
        return df

    def classify_effectiveness(self):
        #  Load model from hf and classify
        pass

    def extract_style_features(self):
        self._extract_basic_features()
        self._extract_transformer_based_features()

    def _extract_transformer_based_features(self):

        transformer_features_path = f"{self.root_path}{self.generated_args_path}/style_features/style_transformer_features_{self.key}.csv"

        print("extracting transformer features")
        
        if Path(transformer_features_path).is_file():
            self.basic_style_df = pd.read_csv(transformer_features_path)
        else:
            # RESET
            print("Initializing style features pipeline")
            pipeline_transformer_features = IESTAFeatureExtractionPipeline(
                save_output=True,
                exec_transformer_based=True,
                argument_col="effective_argument",
                idx_col="idx"
            )
            pipeline_transformer_features.spacy_n_processors = 1
            pipeline_transformer_features.init_and_run()
            pipeline_transformer_features.reset_input_output()
            pipeline_transformer_features.out_path = transformer_features_path
            filter_df = self.data[self.data["success"]]
            pipeline_transformer_features.set_input(filter_df)
            assert len(filter_df[filter_df["effective_argument"] == ""]) == 0
            try:
                pipeline_transformer_features.annotate()
                pipeline_transformer_features.save()
            except Exception as e:
                print(f"An exception occurred while extracting transformer style features {e}")

    def _extract_basic_features(self):
        
        style_features_path = f"{self.root_path}{self.generated_args_path}/style_features/style_features_{self.key}.csv"
        print("extracting style features")

        if Path(style_features_path).is_file():
            self.basic_style_df = pd.read_csv(style_features_path)
        else:
            pipeline_basic_features = IESTAFeatureExtractionPipeline(
                save_output=True,
                exec_transformer_based=False,
                argument_col="effective_argument",
                idx_col="idx"
            )
            pipeline_basic_features.spacy_n_processors = 2
            pipeline_basic_features.init_and_run()
            pipeline_basic_features.reset_input_output()
            pipeline_basic_features.out_path = style_features_path

            pipeline_basic_features.set_input(self.data[self.data["success"]])
            try:
                pipeline_basic_features.annotate()
                pipeline_basic_features.save()
            except Exception as e:
                print(f"An exception occurred while extracting style features {e}")

    def _get_style_features(self):
        print("get_style_features in one df")
        style_features_path = f"{self.root_path}{self.generated_args_path}/style_features/style_features_{self.key}.csv"
        transformer_features_path = f"{self.root_path}{self.generated_args_path}/style_features/style_transformer_features_{self.key}.csv"

        # LIWC
        liwc_fpath = f"{self.root_path}{self.generated_args_path}/style_features/liwc_{self.key}.csv"
        liwc_df = pd.read_csv(liwc_fpath, index_col="idx")
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        liwc_df = liwc_df.select_dtypes(include=numerics)
        liwc_df.drop(columns=["Segment", "round"], inplace=True)
        liwc_df.columns = "liwc_" + liwc_df.columns

        empath_mpqa_df = pd.read_csv(style_features_path)
        transformers_based_features_df = pd.read_csv(transformer_features_path)

        difference = transformers_based_features_df.columns.difference(
            empath_mpqa_df.columns
        )
        feature_df = empath_mpqa_df.merge(
            transformers_based_features_df[difference],
            right_index=True,
            left_index=True,
        )
        feature_df = feature_df.merge(
            liwc_df,
            right_index=True,
            left_index=True,
        )
        return feature_df

    def score_style(self) -> pd.DataFrame:
        self.extract_style_features()
        top_features: pd.DataFrame = get_top_features(self.ideology)
        style_features = self._get_style_features()

        print("creating style feature score...")
        scaler = QuantileTransformer()
        columns_to_normalize = top_features["feature"].values.tolist()
        style_features[columns_to_normalize] = style_features[columns_to_normalize].fillna(0.0)
        style_features[columns_to_normalize] = scaler.fit_transform(
            style_features[columns_to_normalize]
        )

        def _add_score_features(row, ideology, top_features):
            feature_col_name = f"effective ineffective_{ideology}"
            features_score: float = 0.0
            for _, feature_row in top_features.iterrows():
                feature_name = feature_row["feature"]
                features_score = features_score + (
                    row[feature_name] * feature_row[feature_col_name] * 1.0
                )
            row["features_score"] = features_score
            return row

        style_features = style_features.apply(
            _add_score_features,
            axis=1,
            args=(
                self.ideology,
                top_features,
            ),
        )

        #style_features = style_features.sort_values(by=["features_score"], ascending=False)
        return style_features

    def score_morality():
        # like malitiousness
        # Expected: LLM significantly decrease it
        pass

    def score_stance():
        # percent of correct ones
        pass

    def score_toxicity(self, texts, truncate_txt: bool = False):
        toxigen_roberta = pipeline(
            "text-classification",
            model="tomh/toxigen_roberta",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        
        def reformat(
            text_scores,
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
            n_grams.append(tuple(tokens[i: i + n]))

        # Count the frequency of each 3-gram
        frequencies = Counter(n_grams)

        # Sort the 3-grams by frequency
        top_n_grams = sorted(frequencies.items(), key=lambda x: x[1])[-top:]

        # Print or store the results
        return top_n_grams
