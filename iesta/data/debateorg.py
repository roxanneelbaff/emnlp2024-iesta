import pandas as pd
import glob
from iesta import properties
from iesta import utils
import pathlib
from tqdm import tqdm


class Loader:
    DATA_TYPE_USER = "USER"
    DATA_TYPE_DEBATE = "DEBATE"

    def __init__(self):
        self.users_df = None
        self.debates_df = None
        return

    # https://esdurmus.github.io/ddo.html
    def get_users(self):
        self.users_df = pd.read_json(
            properties.DEBATEORG_USERS_JSON_PATH, orient="index"
        )
        self.users_df.index.name = "user_name"
        return self.users_df

    def get_debates(self):
        self.debates_df = pd.read_json(
            properties.DEBATEORG_DEBATES_JSON_PATH, orient="index"
        )
        return self.debates_df

    def get_user_political_parties(self, plot=True):
        parties_count_df = self.get_value_counts_df(
            "party",
            data_type=Loader.DATA_TYPE_USER,
            count_name="user_count",
            index_name="political_party",
            plot=plot,
        )
        return parties_count_df

    def get_user_ideologies(self, plot=True):
        ideologies_count_df = self.get_value_counts_df(
            "political_ideology",
            data_type=Loader.DATA_TYPE_USER,
            count_name="user_count",
            index_name="ideology",
            plot=plot,
        )
        return ideologies_count_df

    def get_debate_categories(self, plot=True):
        categories_count_df = self.get_value_counts_df(
            "category",
            data_type=Loader.DATA_TYPE_DEBATE,
            count_name="debate_count",
            index_name="debate_category",
            plot=plot,
        )
        return categories_count_df

    def get_value_counts_df(
        self, column, data_type, count_name=None, index_name=None, plot=False
    ):
        count_df = (
            self.users_df[column].value_counts().to_frame(name=count_name)
            if data_type == Loader.DATA_TYPE_USER
            else self.debates_df[column]
            .value_counts()
            .to_frame(name=count_name)
        )
        count_df.index.name = index_name
        if plot:
            top = 15 if len(count_df) > 15 else len(count_df)
            count_df.head(top).plot(kind="bar")
        return count_df

    def get_liberal_conservative_users(self):
        lib_cons_users_df = self.users_df[
            (
                self.users_df["political_ideology"]
                == properties.LIBERAL_IDEOLOGY
            )
            | (
                self.users_df["political_ideology"]
                == properties.CONSERVATIVE_IDEOLOGY
            )
        ]
        liberal_cons_user_ids = lib_cons_users_df.index.values.tolist()
        print(
            "# of liberal and conservative users", len(liberal_cons_user_ids)
        )
        return lib_cons_users_df, liberal_cons_user_ids

    def get_debates_w_liberal_or_conservative_paticipants(self):
        _, liberal_cons_user_ids = self.get_liberal_conservative_users()
        df = self.debates_df[
            (self.debates_df["participant_1_name"].isin(liberal_cons_user_ids))
            | (
                self.debates_df["participant_2_name"].isin(
                    liberal_cons_user_ids
                )
            )
        ].copy()
        print(
            "# of debates with liberal OR conservative participants: ", len(df)
        )

        return df

    def _get_user_ideology(self, user):
        ideology = None
        if user not in self.users_df.index.values.tolist():
            ideology = "NOT FOUND"
        else:
            ideology = self.users_df.loc[user]["political_ideology"]
        return ideology

    # returns a flat df for each voter
    # columns: debate_id, category (debate), p{1,2}_name, p{1,2}_ideology, voter_ideology,
    #          p{1,2}_agree_before, p{1,2}_agree_after, p{1,2}_convincing
    def flatten_debate_votes(self):
        result_arr = []
        debates_err_arr = []
        for i, row in tqdm(
            self.debates_df.iterrows(), total=self.debates_df.shape[0]
        ):
            debate_info = {}
            debate_info["debate_id"] = i
            debate_info["category"] = row["category"]
            debate_info["p1_name"] = row["participant_1_name"]
            debate_info["p2_name"] = row["participant_2_name"]

            debate_info["p1_ideology"] = self._get_user_ideology(
                row["participant_1_name"]
            )
            debate_info["p2_ideology"] = self._get_user_ideology(
                row["participant_2_name"]
            )

            if row["number_of_votes"] < 1:
                continue
            for voter in row["votes"]:
                try:
                    debate_voter = debate_info.copy()
                    debate_voter["voter_username"] = voter["user_name"]

                    if (
                        voter["user_name"]
                        not in self.users_df.index.values.tolist()
                    ):
                        continue
                    debate_voter["voter_ideology"] = self._get_user_ideology(
                        voter["user_name"]
                    )

                    p1_votes = voter["votes_map"][row.participant_1_name]
                    p2_votes = voter["votes_map"][row.participant_2_name]

                    debate_voter["p1_agree_before"] = p1_votes[
                        "Agreed with before the debate"
                    ]
                    debate_voter["p1_agree_after"] = p1_votes[
                        "Agreed with after the debate"
                    ]
                    debate_voter["p1_convincing"] = p1_votes[
                        "Made more convincing arguments"
                    ]

                    debate_voter["p2_agree_before"] = p2_votes[
                        "Agreed with before the debate"
                    ]
                    debate_voter["p2_agree_after"] = p2_votes[
                        "Agreed with after the debate"
                    ]
                    debate_voter["p2_convincing"] = p2_votes[
                        "Made more convincing arguments"
                    ]

                    result_arr.append(debate_voter)
                except Exception as e:
                    debates_err_arr.append(i)

                    # print(e)

        result_df = pd.DataFrame(result_arr)
        return result_df, debates_err_arr


@staticmethod
def get_args_w_effect(ideology):
    ideology = ideology.lower()
    args_df = pd.read_parquet(
        properties.DEBATEORG_ARGS_W_EFFECT_PATH.format(ideology)
    )
    return args_df


# The data is already saved under DEBATEORG_ARGS_TEXT_FOLDER_PATH: ../data/debateorg_arguments_txt
def save_args_as_texts(ideology):
    ideology = ideology.lower()
    df = get_args_w_effect(ideology)

    folder_path = properties.DEBATEORG_ARGS_TEXT_FOLDER_PATH.format(ideology)
    utils.create_folder(folder_path)

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        arg = row["argument"].strip()
        if len(arg) == 0 or arg == "forfeit":
            continue

        with open(
            folder_path + "{}_{}.txt".format(ideology, ideology, str(idx)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(arg)


# These paths are used as input for the opinion finder
def save_arg_txt_links(ideology):
    ideology = ideology.lower()
    path_args_txt = str(
        pathlib.Path(properties.DEBATEORG_ARGS_TEXT_FILES).resolve()
    )
    txt_links = glob.glob(path_args_txt.format(ideology))

    links_str = "\n".join(txt_links)

    with open(
        properties.DOCLIST_PATH.format(ideology), "w", encoding="utf-8"
    ) as f:
        f.write(links_str)
