# Files and folders paths

import os

# "C:\Users\elba_ro\Documents\projects\github\conf22-style-transfer"
ROOT_PATH = os.path.join(r"..", "data")
DEBATEORG_USERS_JSON_PATH = os.path.join(ROOT_PATH, "debateorg", "users.json")
DEBATEORG_DEBATES_JSON_PATH = os.path.join(
    ROOT_PATH, "debateorg", "debates.json"
)


FLAT_VOTES_W_EFFECT_FILE = os.path.join(
    ROOT_PATH, "flat_voter_w_effect.parquet"
)
MISSING_FILE = os.path.join(ROOT_PATH, "missing_ideology_debates.parquet")

DEBATEORG_ARGS_W_EFFECT_PATH = os.path.join(
    ROOT_PATH, "{}_debate_arguments_w_effect.parquet"
)
MISSING_ARGS_FILE = os.path.join(ROOT_PATH, "{}_not_found_arguments.parquet")
DEBATEORG_ARGS_TEXT_FOLDER_PATH = os.path.join(
    ROOT_PATH, "debateorg_arguments_txt", "{}", ""
)
DEBATEORG_ARGS_TEXT_FILES = os.path.join(
    ROOT_PATH, "debateorg_arguments_txt", "{}", "*txt"
)

DOCLIST_PATH = os.path.join(ROOT_PATH, "{}_args_paths.doclist")


# Extracted features
FEATURE_LIWC_PATH = os.path.join(
    ROOT_PATH, "extracted_features", "liwc_{}.parquet"
)
RAW_LIWC_PATH = os.path.join(
    ROOT_PATH, "extracted_features", "liwc_original", "liwc_{}.csv"
)

FEATURE_NRC_PATH = os.path.join(
    ROOT_PATH, "extracted_features", "nrc_{}.parquet"
)
FEATURE_MPQA_ARG_PATH = os.path.join(
    ROOT_PATH, "extracted_features", "mpqa_arg_{}.parquet"
)
FEATURE_EMPATH_PATH = os.path.join(
    ROOT_PATH, "extracted_features", "empath_{}.parquet"
)
FEATURE_EMPATH_IDEOLOGY_PATH = os.path.join(
    ROOT_PATH, "extracted_features", "empath_ideology_{}.parquet"
)


# Constants

CONSERVATIVE_IDEOLOGY = "Conservative"
LIBERAL_IDEOLOGY = "Liberal"

IDEOLOLGY_LST = [
    CONSERVATIVE_IDEOLOGY.lower(),
    LIBERAL_IDEOLOGY.lower(),
]

LONG_LIBERAL_TXT_INDX = [
    2472,
    30429,
    30428,
    50796,
    25960,
    207,
    8142,
    46764,
    18639,
    45176,
    19562,
    8136,
]
LONG_CONSERVATIVE_TXT_INDX = [44833, 12198, 44834, 28829, 14753]
