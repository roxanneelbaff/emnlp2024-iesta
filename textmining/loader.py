import glob
import pandas as pd
import numpy as np
import platform
from functools import reduce
import machine_learning
import os


class loader:
    lean_ideologies = [
        "Market Skeptic Republicans",
        "New Era Enterprisers",
        "Disaffected Democrats",
    ]
    extreme_ideologies = [
        "Country First Conservatives",
        "Opportunity Democrats",
        "Solid Liberals",
    ]
    effect_mapping = {
        1: "strongly_challenging",
        2: "challenging",
        3: "no_effect",
        4: "reinforcing",
        5: "empowering",
    }
    abstract_effect_mapping = {
        1: "challenging",
        2: "no_effect",
        3: "reinforcing",
    }

    def __init__(self):
        print('"corpus" is set. It contains the 6000 annotation')
        self.corpus = self.load_corpus()
        self.liberal = None
        self.conservative = None
        self.data_division = None
        self.pars_features_df = None
        self.discourse_adus_df = None
        self.data_division_df = None
        self.df_features = None

    def load_corpus(self):
        root = "/home/cifo3206/Documents/projects/"
        if platform.system() == "Windows":
            root = "C:/Users/elba_ro/eclipse-workspace/"

        path_corpus = (
            root
            + "research-in-progress/arguana/CONLL-18/conll18-news-editorials-quality-data/corpus-webis-editorial-quality-18.json"
        )

        corpus = pd.read_json(path_corpus)

        return corpus

    def load_data_with_features(self):
        if self.df_features is None:
            self.df_features = pd.read_json(
                "data/articles_with_adu_liwc_lexicons.json", orient="records"
            )
            self.df_features.set_index("idx", inplace=True)
        return self.df_features

    def get_normalized_training_data(self, normalizing_method="sqrt"):
        self.load_data_with_features()
        df_train = self.df_features[self.df_features["split_label"] == "train"]

        df_train, _ = machine_learning.clip_outliers(
            df_train, df_test=None, lower_percentile=1, upper_percentile=99
        )

        X_train_df = df_train.select_dtypes(include=[np.number])

        y_train_df = df_train[["liberal_majority", "conservative_majority"]]
        X_train_df, _ = machine_learning.normalize(
            X_train_df, None, normalizing_method="sqrt"
        )

        result_df = X_train_df.join(y_train_df)
        return result_df

    @staticmethod
    def _apply_set_split_label(row, df_features):
        aid = int(row.name.split("-")[0])
        if aid not in df_features.index:
            row["dismiss"] = "Yes"
        else:
            row["dismiss"] = "No"
            discourse_row = df_features.loc[aid]
            row["split_label"] = discourse_row["split_label"]
            row["liberal_majority"] = discourse_row["liberal_majority"]
            row["conservative_majority"] = discourse_row[
                "conservative_majority"
            ]

        return row

    @staticmethod
    def _apply_discourse_level(row, df):
        df = df[df["id"] == row["id"]]
        end_par = df["order-in-discourse"].max()
        if row["order-in-discourse"] == 1 or row["order-in-discourse"] == 2:
            row["discourse_level"] = "lead"
        elif row["order-in-discourse"] == end_par:
            row["discourse_level"] = "end"
        else:
            row["discourse_level"] = "body"

        return row

    @staticmethod
    def load_discourse_liwc():
        liwc_df = pd.read_json("data/articles_features.json", orient="records")
        liwc_df.set_index("article_id", inplace=True)
        liwc_cols = [c for c in liwc_df.columns if c.startswith("liwc")]
        liwc_df = liwc_df[liwc_cols]
        return liwc_df

    def load_discourse_adus(self):
        # discourse_features_df = pd.read_json('data/articles_features.json', orient='records')
        # discourse_features_df.set_index('id', inplace=True)
        # discourse_features_df = discourse_features_df.apply(loader.apply_set_split_label, args=(self.data_division,), axis=1)

        discourse_adus_df = pd.read_json(
            "data/discourse_adus_annotations.json", orient="records"
        )
        discourse_adus_df.set_index("id", inplace=True)

        self.discourse_adus_df = discourse_adus_df
        print('"discourse_adus_df" is set with id of article id without txt')
        return self.discourse_adus_df

    def load_saved_pars_df(self):
        self.load_data_with_features()
        pars_features_df = pd.read_json(
            "data/paragraphs_with_adu_liwc_lexicons.json", orient="records"
        )
        pars_features_df.set_index("paragraph-id", inplace=True)

        if "split_label" not in pars_features_df.columns.values:
            print("setting split_label")
            pars_features_df = pars_features_df.apply(
                loader._apply_set_split_label, args=(self.df_features,), axis=1
            )
        if "discourse_level" not in pars_features_df.columns.values:
            print("setting discourse_level")
            pars_features_df = pars_features_df.apply(
                loader._apply_discourse_level, args=(pars_features_df,), axis=1
            )
        self.pars_features_df = pars_features_df
        return self.pars_features_df

    def load_paragraphs_with_features(self, content=False):
        self.load_data_with_features()
        pars_features_df = pd.read_json(
            "data/pars_features.json", orient="records"
        )
        pars_features_df.set_index("paragraph-id", inplace=True)
        print(pars_features_df.columns)

        pars_adus_df = pd.read_json(
            "data/paragraphs_adus_annotations.json", orient="records"
        )
        pars_adus_df.set_index("id", inplace=True)
        pars_features_df = pars_features_df.join(pars_adus_df)
        print(
            '"pars_features_df" is set. Each row is a paragraph of the article with features'
        )
        pars_features_df = pars_features_df[
            pars_features_df["dismiss"] == "No"
        ]
        keep_cols = [
            c
            for c in pars_features_df.columns.values
            if c.startswith("liwc_") or c.startswith("adu_")
        ]
        keep_cols.extend(
            [
                "id",
                "index",
                "conservative_majority",
                "liberal_majority",
                "order-in-discourse",
                "mpqa_par_sentences_obj",
                "mpqa_par_sentences_subj",
            ]
        )
        pars_features_df = pars_features_df[keep_cols]
        pars_features_df = pars_features_df.apply(
            loader._apply_set_split_label, args=(self.df_features,), axis=1
        )
        pars_features_df = pars_features_df.apply(
            loader._apply_discourse_level, args=(pars_features_df,), axis=1
        )
        if content:
            df_ = pd.read_json(
                "../acl2019/data/pars_liwc.json", orient="records"
            )
            df_.set_index("paragraph-id", inplace=True)
            df_ = df_[["content"]]
            pars_features_df = pars_features_df.join(df_)
        self.pars_features_df = pars_features_df
        return self.pars_features_df

    # helper method to get content
    @staticmethod
    def apply_get_article_content(row):
        article_id = row["article_id"]
        path = "corpus/{}".format(article_id)
        text = ""  # get text
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            row["content"] = text.strip()
        # row = loader.clean_content_ending(row, pars_df)
        return row

    # @staticmethod
    # def clean_content_ending(row, pars_df):
    #    article_content = row['content']
    #    end_pars = pars_df[pars_df['discourse_level'] == 'end'][['id', 'content']]
    #    clean_end_content = end_pars[end_pars['id'] == int(row['article_id'].replace(".txt", ""))]['content'].values[0]
    #    idx_ = article_content.find(clean_end_content) +len(clean_end_content)
    #    #print('removed: ', row['content'][idx_:])
    #    row['removed'] = row['content'][idx_:]
    #    row['content'] = row['content'][:idx_]
    #    return row

    @staticmethod
    def apply_add_majority(row):
        effect_map = {0: "challenging", 1: "no_effect", 2: "reinforcing"}
        results = [row["challenging"], row["no_effect"], row["reinforcing"]]
        effect_num = np.argmax(
            results
        )  # if equal, returns 0 which is considered challenging
        row["majority"] = effect_map[effect_num]
        row["majority_int"] = effect_num
        return row

    ## helper:
    @staticmethod
    def apply_add_ids(row):
        row["ids"] = int(row["article_id"].replace(".txt", ""))
        return row

    @staticmethod
    def apply_set_idx(row):
        row["idx"] = (
            int(row["ids"].split(",")[0])
            if ("," in str(row["ids"]))
            else int(row["ids"])
        )
        return row

    @staticmethod
    def sum_str(series):
        return reduce(lambda x, y: str(x) + "," + str(y), series)

    def get_article_dfs_per_ideology(
        self, ideology="political_pole", include_content=False
    ):
        # if self.data_division is None:
        #    print("train/test/validation set were not specified - Setting train test to 80\% 20\% ")
        #    self.get_train_test_data(train_percent = 0.8, has_validation_data = False,
        #                                          add_to_corpus = True)
        result = {}
        for ideology, ideology_df in self.corpus.groupby([ideology]):
            df = pd.DataFrame(
                columns=[
                    "article_id",
                    "challenging",
                    "no_effect",
                    "reinforcing",
                    "split_label",
                ]
            )
            for aid, ideology_df in ideology_df.groupby(["article_id"]):
                vals = ideology_df["effect_abstracted"].value_counts()
                row = {}
                row["article_id"] = aid
                for k in vals.keys():
                    row[self.abstract_effect_mapping[k]] = vals[k]

                # article_id_int = int((row['article_id'].split('.')[0]))
                # for k in self.data_division.keys():
                #    if article_id_int in self.data_division[k]:
                #        row['split_label'] = k
                #        break
                df = df.append(row, ignore_index=True)
                df = df.fillna(0)

                if include_content:
                    # if self.pars_features_df is None:
                    #    self.load_paragraphs_with_features()
                    df = df.apply(loader.apply_get_article_content, axis=1)
            print(
                "articles dataframe for ideology {} was created".format(
                    ideology
                )
            )
            print("The id of the df is the article id without txt")
            df = df.apply(loader.apply_add_ids, axis=1)
            df = df.groupby(["content"], as_index=False).agg(
                {
                    "challenging": "sum",
                    "no_effect": "sum",
                    "reinforcing": "sum",
                    "ids": loader.sum_str,
                }
            )
            df = df.apply(loader.apply_set_idx, axis=1)
            df = df.apply(loader.apply_add_majority, axis=1)
            self.get_train_test_data(
                train_percent=0.8,
                has_validation_data=False,
                add_to_corpus=True,
                article_ids=list(df["idx"].values),
            )

            df.set_index("idx", inplace=True)
            print("length of self.df: ", len(df))
            print(
                "length of self.data_division_df: ", len(self.data_division_df)
            )
            df = df.join(self.data_division_df)
            print("length of self.df: ", len(df))
            result[ideology] = df
        return result

    def get_explanation_subcorpus(self, save=False):
        self.corpus_explanation = self.corpus[
            ["annotator_id", "article_id", "effect", "explanation"]
        ]
        if save:
            fname = "corpus/corpus_explanation.csv"
            if os.path.isfile(fname):
                print(
                    "Warning! File already exists. Deleting and recreating:",
                    fname,
                )
                os.remove(fname)
                self.corpus_explanation.to_csv(
                    "corpus/corpus_explanation.csv", encoding="utf-8"
                )
        return self.corpus_explanation

    def get_ideologies_dfs(self):
        print('"liberal" and "conservative" dataframes are set')
        self.liberal = self.corpus[self.corpus["political_pole"] == "liberal"]

        self.conservative = self.corpus[
            self.corpus["political_pole"] == "conservative"
        ]
        return self.liberal, self.conservative

    # def get_ideologies_hotencoded_effect():
    @staticmethod
    def apply_ideology_intensity(row):
        ideology = row["political_typology"]

        if ideology in loader.lean_ideologies:
            row["ideology_intensity"] = "lean"
        elif ideology in loader.extreme_ideologies:
            row["ideology_intensity"] = "extreme"
        else:
            print(ideology, " not specified")
        return row

    def add_ideology_intensity(self):
        print('"corpus" is set with ideology_intensity: "extreme" and "lean"')

        self.corpus = self.corpus.apply(
            loader.apply_ideology_intensity, axis=1
        )
        return self.corpus

    def get_train_test_data(
        self,
        train_percent=0.8,
        has_validation_data=True,
        add_to_corpus=False,
        article_ids=None,
    ):
        article_ids = (
            [
                int((x.split(".")[0]))
                for x in self.corpus["article_id"].unique()
            ]
            if (article_ids is None)
            else article_ids
        )

        # dismiss duplicates

        article_ids.sort()
        print("total con:", len(article_ids))
        # TRAINGING
        training_num = round(len(article_ids) * train_percent)

        train_article_ids = article_ids[0:training_num]

        print("rounded TRAINGING data: ", len(train_article_ids))

        # define validation and test size
        test_percent = (
            ((1 - train_percent) / 2)
            if has_validation_data
            else 1 - train_percent
        )
        validate_percent = test_percent if has_validation_data else 0

        validate_article_ids = article_ids[
            training_num : training_num
            + round(len(article_ids) * validate_percent)
        ]
        test_article_ids = article_ids[
            -round(len(article_ids) * test_percent) :
        ]

        print("rounded Validation data: ", len(validate_article_ids))
        print("rounded Test data: ", len(test_article_ids))

        self.data_division = {
            "train": train_article_ids,
            "test": test_article_ids,
        }

        df_train = pd.DataFrame(
            {"idx": train_article_ids, "split_label": "train"}
        )
        df_test = pd.DataFrame(
            {"idx": test_article_ids, "split_label": "test"}
        )
        df = df_train.append(df_test)

        if has_validation_data:
            self.data_division["validate"] = validate_article_ids
            df_validate = pd.DataFrame(
                {"idx": test_article_ids, "split_label": "validation"}
            )
            df = df.append(df_validate)

        if add_to_corpus:
            train_test_dict = self.get_train_test_data(
                train_percent=train_percent,
                has_validation_data=has_validation_data,
                add_to_corpus=False,
            )
            print('"corpus" is set with split_label: "train" "test"')
            self.corpus = self.corpus.apply(
                loader.apply_test_train_split, args=(train_test_dict,), axis=1
            )
        print(
            '"data_division" is set as dict with keys ',
            self.data_division.keys(),
        )
        df.set_index("idx", inplace=True)
        self.data_division_df = df
        return self.data_division

    # def get_ideologies_hotencoded_effect():
    @staticmethod
    def apply_test_train_split(row, test_train_dict):
        article_id_int = int((row["article_id"].split(".")[0]))

        for k in test_train_dict.keys():
            if article_id_int in test_train_dict[k]:
                row["split_label"] = k
                break

        return row
