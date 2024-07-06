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
import scipy.stats as stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

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

        self.eval_file_format: str = (
            f"{self.root_path}{self.generated_args_path}/eval/eval_"+"{}{}"
        )

        print(f"1. postprocessing the generated arguments {self.filename}...")
        self.data = process_llm_generated_args(self.filename,
                                               root=self.root_path)

        print("fetching original ineffective arguments...")
        self.original_data_df = self.get_test_data()

        print(f"2. Calculating the toxicity scores for Ineffective arguments - existing cols {self.original_data_df.columns.tolist()}...")
        #print(f"{type(self.original_data_df['text'].values.tolist())}")
        
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
            print("Toxicity score already calculated for INPUT (ineffective data)")  

        print("-- merging original argument info with generated one")
        self.merged_df = pd.merge(
            self.original_data_df[[
                    "category",
                    "round",
                    "debate_id",
                    "idx",
                    "toxic",
                    "text"
                ]],
            self.data,
        
            how="inner",
            on="idx",
        )
        print(f"-- merged data length: 3500 -> {len(self.merged_df)} --> loss: {3500-len(self.merged_df)}")

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
        print(" ### Stats ### \n3.a adding has_ideology_prompt")
        self.merged_df = self.merged_df.apply(add_has_ideology_prompt, axis=1)

        print("3.b setting count_success_per_prompt_df")
        self.count_success_per_prompt_df = self.get_cross_counts(
            idx="prompt", col="success"
        )

        print("3.c setting count_success_per_category_df")
        self.count_success_per_category_df = self.get_cross_counts(
            idx="category", col="success"
        )

        print("3.d setting count_type_per_prompt_df")
        self.count_type_per_prompt_df = self.get_cross_counts(
            idx="prompt", col="type"
        )

        print("3.e setting count_success_per_ideology_prompt_df")
        self.count_success_per_ideology_prompt_df = self.get_cross_counts(
            idx="has_ideology_prompt", col="success"
        )

        print("3.f setting count_success_per_toxic_df")
        self.count_success_per_toxic_df = self.get_cross_counts(
            idx="toxic_str", col="success"
        )

        print("3.g setting top_ngrams_count_failed_response n=10, top=30")
        self.top_ngrams_count_failed_response = (
            self.analyze_failed_output_w_ngrams(n=10, top=30)
        )

        print("3.h setting failed_ratio")
        self.failed_ratio = (
            len(self.data[~self.data["success"]]) * 100 / float(len(self.data))
        )
        
        self.total = len(self.data)
        self.failed_num = len(self.data[~self.data["success"]])
        print(f"-- failed_ratio: failed_count/totalnum --> {round(self.failed_ratio, 2)}: {self.failed_num} / {len(self.data)}")

        print("3.i setting corr_toxicity_no_response")
        self.corr_toxicity_no_response = self.calc_corr_toxicity_no_response()

        self.successful_df = self.merged_df[self.merged_df["success"]]

        print("4. ### Scoring style ### ")
        self.style_score_df = self.score_style()

        all_w_style_df = pd.merge(
            self.merged_df,
            self.style_score_df,
            left_index=True,
            right_index=True,
            how="inner",)
        print(f"style_df len: {len(all_w_style_df)} and success_df {len(self.successful_df)}")
        self.ineffective_style_score_df = self.score_style(is_for_ineffective=True)
        # CALCULATE SIGNIFICANCY
        # within model: to reveal the importance of a prompt
        # model vs. ineffective
        self.style_score_sign = []
        for p, df_ in all_w_style_df.groupby("prompt"):
            generated_vals = df_[df_["success"]]["features_score"].values.tolist()
            input_vals = self.ineffective_style_score_df["features_score"].values.tolist()
            res = self.significance([generated_vals,
                                      input_vals])
            res["prompt"] = p
            res["ineffective_median"] = np.median(generated_vals)
            res["generated_median"] = np.median(input_vals)
            # TODO: ADD Top 5 significant features?
            self.style_score_sign.append(res)
        self.style_score_sign = pd.DataFrame(self.style_score_sign)

        # FOR SUCCESSFUL RESPONSES
        effectiveness_fname = self.eval_file_format.format(self.key, "_effectiveness.csv")
        # print(effectiveness_fname)
        original_effectiveness_fname = self.eval_file_format.format(f"original_ineffective_{self.ideology}", "_effectiveness.csv")
        if Path(effectiveness_fname).is_file():
            print("Effectiveness file found")
            self.effectiveness_scores = pd.read_csv(effectiveness_fname)
            self.successful_df["effective"] = self.effectiveness_scores["effective"]
        else:
            
            print("Effectiveness for succssful generated")
            self.effectiveness_scores = self.classify_effectiveness(
                self.successful_df["effective_argument"].values.tolist()
            )
            self.successful_df["effective"] = self.effectiveness_scores["effective"]
            self.effectiveness_scores.to_csv(effectiveness_fname)

        if Path(original_effectiveness_fname).is_file():
            print("Effectiveness for original found")
            self.effectiveness_original_scores = pd.read_csv(original_effectiveness_fname)
            
        else:
            print("Effectiveness for original")
            self.effectiveness_original_scores = self.classify_effectiveness(
                self.original_data_df["text"].values.tolist()
            )
            self.effectiveness_original_scores.to_csv(original_effectiveness_fname)

        self.eff_score_sign = []
        for p, df_ in self.successful_df.groupby("prompt"):
            generated_eff_vals = df_[~df_["effective"].isna()]["effective"].values.tolist()
            origin_effectiveness_vals = self.effectiveness_original_scores["effective"].values.tolist()
            res = self.significance([generated_eff_vals,
                                      origin_effectiveness_vals])
            res["prompt"] = p
            
            res["ineffective_median"] = np.median(origin_effectiveness_vals)
            res["generated_median"] = np.median(generated_eff_vals)
            self.eff_score_sign.append(res)
        self.eff_score_sign = pd.DataFrame(self.eff_score_sign)

        print("***** END ****")

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
    
    def calc_corr_two_binary(
        self, col1="has_ideology_prompt", col2="success"
    ):
        result = []

        def calc_corr(df_):
            arr1 = [
                (1 if x else 0) for x in df_[col1].values.tolist()
            ]
            arr1_np = np.array(arr1)
            arr2 = [
                (1 if x else 0) for x in df_[col2].values.tolist()
            ]
            arr2_np = np.array(arr2)

            corr, p_value = pointbiserialr(
                arr1_np, arr2_np
            )
            return corr, p_value

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

    def classify_effectiveness(self, texts):
        effect_classifier = pipeline("text-classification",
                                     f"notaphoenix/{self.ideology}_argument_classifer",
                                     top_k=None,
                                     truncation=True,
                                     max_length=2048,
                                     device=0,
        )

        def reformat(
            text_scores,
        ):  # [{'label': 'ineffective', 'score': 0.5774604678153992},{'label': 'effective', 'score': 0.42253953218460083}]
  
            return {x["label"]: x["score"] for x in text_scores}

        scores_ = effect_classifier([str(txt) for txt in texts])
        scores_ = [reformat(x) for x in scores_]

        return pd.DataFrame(scores_)
       

    def extract_style_features(self, rerun_style_extraction: bool = False):
        self._extract_basic_features(rerun_style_extraction)
        self._extract_transformer_based_features(rerun_style_extraction)


    def _extract_ineffective_base(self):
        style_features_path = f"{self.root_path}{self.generated_args_path}/style_features/ineffective_{self.ideology}.parquet"
        print("extracting style features")

        if not Path(style_features_path).is_file():
            pipeline_basic_features = IESTAFeatureExtractionPipeline(
                save_output=True,
                exec_transformer_based=False,
                argument_col="text",
                idx_col=None, #"idx"
            )
            pipeline_basic_features.spacy_n_processors = 2
            pipeline_basic_features.init_and_run()
            pipeline_basic_features.reset_input_output()
            pipeline_basic_features.out_path = style_features_path
            filter_df = self.original_data_df.copy()
            filter_df["text"].fillna("", inplace=True)
            filter_df = filter_df[~(filter_df["text"].str.fullmatch(""))]
            pipeline_basic_features.set_input(filter_df)

            print(len(filter_df[filter_df["text"] == ""]),
                  " should be 0)")
            try:
                print("annotating...")
                pipeline_basic_features.annotate()
                print("saving...")
                pipeline_basic_features.save()
            except Exception as e:
                print(f"An exception occurred while extracting style features {e}")
        self.ineffective_basic_style_df = pd.read_parquet(style_features_path)

    def _extract_ineffective_trans_features(self):

        transformer_features_path = f"{self.root_path}{self.generated_args_path}/style_features/ineffective_transformer_{self.ideology}.parquet"

        print("extracting transformer features")
        
        if Path(transformer_features_path).is_file():
            print("ineffective trans has features")
            self.ineffective_transformer_style_df = pd.read_parquet(transformer_features_path)
        else:
            # RESET
            print("Initializing style features pipeline")
            pipeline_transformer_features = IESTAFeatureExtractionPipeline(
                save_output=True,
                exec_transformer_based=True,
                argument_col="text",
                idx_col=None, #"idx"
            )
            pipeline_transformer_features.spacy_n_processors = 1
            pipeline_transformer_features.init_and_run()
            pipeline_transformer_features.reset_input_output()
            pipeline_transformer_features.out_path = transformer_features_path
            filter_df = self.original_data_df.copy()
            filter_df["text"].fillna("", inplace=True)
            filter_df = filter_df[filter_df["text"] != ""]
            pipeline_transformer_features.set_input(filter_df)
            
            #print(len(filter_df[filter_df["text"] == ""]),
            #      " should be 0)")
            try:
                print("annotating...")
                pipeline_transformer_features.annotate()
                print("saving...")
                pipeline_transformer_features.save()
            except Exception as e:
                print(f"An exception occurred while extracting transformer style features {e}")
        self.ineffective_transformer_style_df = pd.read_parquet(transformer_features_path)
    
    def _extract_transformer_based_features(self, rerun_style_extraction: bool = False):

        transformer_features_path = f"{self.root_path}{self.generated_args_path}/style_features/style_transformer_features_{self.key}.parquet"

        print("extracting transformer features")
        
        if Path(transformer_features_path).is_file() and not rerun_style_extraction:
            self.transformer_style_df = pd.read_parquet(transformer_features_path)
        else:
            # RESET
            print("Initializing style features pipeline")
            pipeline_transformer_features = IESTAFeatureExtractionPipeline(
                save_output=True,
                exec_transformer_based=True,
                argument_col="effective_argument",
                idx_col=None, #"idx"
            )
            pipeline_transformer_features.spacy_n_processors = 1
            pipeline_transformer_features.init_and_run()
            pipeline_transformer_features.reset_input_output()
            pipeline_transformer_features.out_path = transformer_features_path
            filter_df = self.data[self.data["success"]].copy()
            filter_df["effective_argument"].fillna("", inplace=True)
            filter_df = filter_df[~(filter_df["effective_argument"].str.fullmatch(""))]
            pipeline_transformer_features.set_input(filter_df)
            
            print(len(filter_df[filter_df["effective_argument"] == ""]),
                  " should be 0)")
            try:
                print("annotating...")
                pipeline_transformer_features.annotate()
                print("saving...")
                pipeline_transformer_features.save()
            except Exception as e:
                print(f"An exception occurred while extracting transformer style features {e}")
        self.transformer_style_df = pd.read_parquet(transformer_features_path)

    def _extract_basic_features(self, rerun_style_extraction: bool = False):
        
        style_features_path = f"{self.root_path}{self.generated_args_path}/style_features/style_features_{self.key}.parquet"
        print("extracting style features")

        if Path(style_features_path).is_file() and not rerun_style_extraction:
            self.basic_style_df = pd.read_parquet(style_features_path)
        else:
            filter_df = self.data[self.data["success"]].copy()
            filter_df["effective_argument"].fillna("", inplace=True)
            filter_df = filter_df[~(filter_df["effective_argument"].str.fullmatch(""))].copy()
            assert len(filter_df) > 0, "DF IS EMPTY line 506"
            
            pipeline_basic_features = IESTAFeatureExtractionPipeline(
                save_output=True,
                exec_transformer_based=False,
                argument_col="effective_argument",
                idx_col=None, #"idx"
            )
            pipeline_basic_features.spacy_n_processors = 2
            
            pipeline_basic_features.reset_input_output()
            pipeline_basic_features.out_path = style_features_path
            #print("empty eff arg should not exists --> 0?", len(filter_df[filter_df["effective_argument"] == ""] ) == 0)
            pipeline_basic_features.set_input(filter_df)
            pipeline_basic_features.init_and_run()

            #try:
            print("annotating...")
            pipeline_basic_features.annotate()
            #print(pipeline_basic_features.out_df[:10])
            print("saving...")
            pipeline_basic_features.save()
            #print(pipeline_basic_features.out_df[:10])
            print(f"Saved file length: {len(pipeline_basic_features.out_df)} and {len(pd.read_parquet(pipeline_basic_features.out_path))}")
        #except Exception as e:
            #    print(f"An exception occurred while extracting style features {e}")
        self.basic_style_df = pd.read_parquet(style_features_path)

    def _get_style_features(self, is_for_ineffective=False):
        print("get_style_features in one df")
        style_features_path = f"{self.root_path}{self.generated_args_path}/style_features/style_features_{self.key}.parquet"
        transformer_features_path = f"{self.root_path}{self.generated_args_path}/style_features/style_transformer_features_{self.key}.parquet"
        if is_for_ineffective:
            style_features_path = f"{self.root_path}{self.generated_args_path}/style_features/ineffective_{self.ideology}.parquet"
            transformer_features_path = f"{self.root_path}{self.generated_args_path}/style_features/ineffective_transformer_{self.ideology}.parquet"

        # LIWC
        liwc_fpath = f"{self.root_path}{self.generated_args_path}/style_features/liwc_{self.key}.csv"
        if is_for_ineffective:
            liwc_fpath = f"{self.root_path}{self.generated_args_path}/style_features/liwc_test_{self.ideology}_ineffective.csv"
        liwc_df = pd.read_csv(liwc_fpath, index_col="idx")
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        liwc_df = liwc_df.select_dtypes(include=numerics)
        if "Segment" in liwc_df.columns.tolist():
            liwc_df.drop(columns=["Segment"], inplace=True)

        liwc_df.columns = "liwc_" + liwc_df.columns

        empath_mpqa_df = pd.read_parquet(style_features_path)
        print(empath_mpqa_df.columns.tolist())
        empath_mpqa_df.set_index('input_id', inplace=True)
        if not is_for_ineffective:
            liwc_df.set_index('liwc_Unnamed: 0', inplace=True)

        transformers_based_features_df = pd.read_parquet(transformer_features_path)
        transformers_based_features_df.set_index('input_id', inplace=True)

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
        print(f"Feature df should big {len(feature_df)}")
        return feature_df

    def score_style(self, is_for_ineffective=False,
                    rerun_style_extraction: bool = False) -> pd.DataFrame:
        self.extract_style_features(rerun_style_extraction)
        top_features: pd.DataFrame = get_top_features(self.ideology, root=self.root_path)
        print(f"top feature {len(top_features)}")
        style_features = self._get_style_features(is_for_ineffective)
        print(f"style_features {len(style_features)}")
        print("creating style feature score...")
        scaler = QuantileTransformer(output_distribution='uniform',
                                     n_quantiles=len(style_features))
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
            features_score = features_score/float(len(top_features))
            row["features_score"] = features_score
            return row
        style_features = style_features.fillna(0)
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
    
    def significance(self, groups:list[list]):
        is_all_normal = True
        try:
            for v in groups:
                if stats.shapiro(v)[1] < 0.05:  # not normal
                    is_all_normal = False
                    break
        except Warning:
            is_all_normal = False

        # LEVENE FOR HOMOGENEITY
        is_homogeneous = False
        if is_all_normal:
            is_homogeneous = stats.levene(*groups)[1] >= 0.05
        stat, p_val = (
            stats.ttest_ind(groups[0], groups[1])
            if (is_all_normal and is_homogeneous)
            else stats.mannwhitneyu(
                groups[0],
                groups[1],
                alternative="two-sided"
                )
                )
        res = {"stat": stat, "p_value": p_val}
        res["test"] = "ttest_ind" if (is_all_normal and is_homogeneous) else "mannwhitneyu"
        return res
