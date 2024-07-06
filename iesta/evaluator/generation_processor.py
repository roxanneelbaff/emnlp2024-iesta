from pathlib import Path
import pandas as pd
from tqdm import tqdm
from glob import glob

import os
## Helpers


def _fetch_all_experiments(
    root="../", path_pattern: str = "data/llms_out/new"
):
    path_pattern = os.path.join(root, path_pattern, "*.jsonl")
    print(path_pattern)
    return [
        os.path.basename(x).replace(".jsonl", "") for x in glob(path_pattern)
    ]


def find_all(a_str, sub):
    start = 0
    res = []
    while start < len(a_str):
        start = a_str.find(sub, start)

        if start == -1:
            return res

        res.append(start)
        start = start + len(sub)  # use start += 1 to find overlapping matches
    return res


def get_between_chars(str_, char_: str = '"'):
    try:
        indices = find_all(str_, char_)

        if len(indices) % 2 != 0:
            str_ = str_[indices[0] + len(char_) :]

        else:
            str_ = str_[(indices[0] + len(char_)) : indices[-1]]
    except Exception:
        return None
    return str_


def _get_final_indices(root, ideology):
    if root != "":
        root = root if root[-1] == "/" else root + "/"
    indices_path = f"{root}data/out/{ideology}_idx.csv"

    preset_indices = []
    print(f"{indices_path}")
    assert Path(indices_path).is_file()
    print("original INDICESS found")
    preset_indices = pd.read_csv(indices_path)["idx"].values.tolist()[:500]
    return preset_indices


# Main Function
def process_llm_generated_args(path, root="../", force_reclean=False):
    text_keys = [
        "base",
        "content",
        "content_style",
        "content_style_ideology",
        "style",
        "style_ideology",
        "ideology",
    ]

    all_experiments = _fetch_all_experiments(root=root)
    experiment = path.split("/")[-1].replace(".jsonl", "")
    ideology = experiment.split("_")[0]
    indices = _get_final_indices(root, ideology)

    df = pd.read_json(path_or_buf=path, lines=True)
    df = df.drop_duplicates(subset=["idx"], keep="last")
    df = df[df["idx"].isin(indices)]
    # df = df.drop("original_text", axis=1)

    out = f"{root}data/llms_out/new/processed/{experiment}_processed.csv"
    if Path(out).is_file() and not force_reclean:
        return pd.read_csv(out)

    # model_type = experiment.split("_")[1]
    ideology = experiment.split("_")[0]

    exceptions_dict = {k: [] for k in all_experiments}

    exceptions_dict["liberal_chatgpt_0shot"] = [
        44708,
        35204,
        3176,
        42440,
        34248,
        51598,
        45614,
        38169,
        16602,
        3259,
        252,
    ]
    exceptions_dict["conservative_chatgpt_0shot"] = [
        79903,
        17012,
        66698,
        824,
        43599,
        46682,
        78779,
        78643,
        58514,
        43596,
        25775,
        32238,
        61381,
    ]
    exceptions_dict["liberal_chatgpt_1shot"] = [252, 20310, 18397, 9161, 45614]
    exceptions_dict["conservative_chatgpt_1shot"] = [
        79903,
        66698,
        40673,
        23110,
        13417,
        46682,
        78778,
        78643,
        11374,
        43596,
        9181,
        49307,
        48861,
    ]  # 10968 "Vote for me for bacon and strippers"

    dismiss_dict = {k: {} for k in all_experiments}
    dismiss_dict["conservative_llamav2_0shot"] = {
        2323: ["style"],
        10968: ["content"],
        10740: [
            "ideology",
            "style_ideology",
            "content_style_ideology",
            "content_style",
            "content",
            "base",
        ],
        24053: ["content_style"],
        36097: ["base", "ideology"],
        77305: ["content_style_ideology"],
        80076: ["style_ideology"],
        52539: ["content_style_ideology"],
        64942: ["content_style_ideology"],
    }

    debug_idx = []
    correct = 7 * len(df)
    processing_logs = []

    phrase_delimiters = [
        f"Effective Argument for Readers with {ideology} Ideology:".lower(),
        f"paraphrased Argument for Readers with {ideology} Ideology:".lower(),
        "Effective argument:".lower(),
        "paraphrased argument:".lower(),
        "Here is a paraphrased version of the ineffective argument:".lower(),
        "Here is a possible paraphrased argument that maintains the original length and stance of the original argument:".lower(),
        "Here is my paraphrased version of the argument:".lower(),
        "Here is my paraphrased version:".lower(),
        "Here is a paraphrased version of the argument:".lower(),
        "Effective:".lower(),
    ]

    no_response = [
        "I cannot comply with your request",
        "I cannot fulfill your request",
        "I cannot provide a paraphrased argument",
        "The argument presented is ineffective",
    ]
    format_type = ""
    for idx, row in tqdm(df.iterrows()):
        for col in text_keys:
            exception = ""
            refuse_to_respond = False
            # TO DISMISS
            if (
                idx in dismiss_dict[experiment].keys()
                and col in dismiss_dict[experiment][idx]
            ):
                success = False
                processing_logs.append(
                    {
                        "idx": row["idx"],
                        "prompt": col,
                        "ineffective_argument": row["text"],
                        "generated": row[col],
                        "effective_argument": "",
                        "cannot_paraphrase": row[col].find(
                            "I cannot provide a paraphrased argument"
                        )
                        > -1,
                        "type": "DISMISSED",  # The argument is between " " or backticks
                        "flag": False,
                        "success": success,
                        "incontent_ideology_mentioned": False > -1
                        and row["original_text"].lower().find(f"{ideology}")
                        < 0,
                        "ideology_mentioned": row[col]
                        .lower()
                        .find(f"{ideology}")
                        > -1
                        and row["original_text"].lower().find(f"{ideology}")
                        < 0,
                        "exception": "Exceptionally dismissed",
                        "llm_out_msge": "",
                    }
                )
                continue
            success = False
            # flag = False
            delimiter_num = 0

            # Phrase delimiter
            format_type = "Not between delimiters and Not rephrased"
            phrase_delimiters_dic = {
                k: row[col].lower().find(k) for k in phrase_delimiters
            }
            latest_phrase_delim = max(
                phrase_delimiters_dic.items(), key=lambda x: x[1]
            )[0]
            paraphras_idx = phrase_delimiters_dic[latest_phrase_delim]
            if paraphras_idx > -1:
                sub = row[col][
                    paraphras_idx + len(latest_phrase_delim) :
                ].strip()
                success = True
                format_type = "between a phrase delimiter"
                delimiter_num = len(latest_phrase_delim)

            # Double quotes
            if not success:
                delimiter_num = 2
                sub = get_between_chars(row[col])
                format_type = (
                    "between quotations" if sub is not None else format_type
                )
                success = sub is not None

            # Sticks
            if not success:
                # if model_type == "llamav2":
                sub = get_between_chars(
                    row[col].replace("´´´´", "```"), char_="```"
                )
                delimiter_num = 6 if sub is not None else delimiter_num
                format_type = (
                    "between sticks" if sub is not None else format_type
                )
                success = sub is not None

            if not success:
                # Search
                format_type = "not between quotations/sticks/phrases"
                sub = row[col]
                for nr in no_response:
                    if sub.lower().find(nr.lower()):
                        refuse_to_respond = True
                        format_type = "refused to respond"
                        success = False
                if not refuse_to_respond:
                    print("!!!!!!! Not by quotes, not sticks and not phrases")
                    print(sub)

            # Exceptional handling - some unmapped quotes
            exceptions = exceptions_dict[experiment]
            if row["idx"] in exceptions:
                success = True
                exception = "exceptionally combine all"
            cleaned_arg = row[col] if row["idx"] in exceptions else sub
            if (
                row["idx"] == 18397
                and experiment == "liberal_chatgpt_1shot"
                and col == "ideology"
            ):
                cleaned_arg = sub
                success = True
                exception = "exceptionally combine all"

            if len(cleaned_arg) < (len(row[col]) - delimiter_num):
                i = row[col].find(cleaned_arg)
                # success = True
                flag_ = True
                # llm_out_msge = (
                #    f"{row[col][0:i]}...{row[col][len(cleaned_arg)+i: ]}"
                # )
                # print(
                #    f"######### {row['idx']} generated text for {col}"
                #    # f"{sub} -\n ****kept:*** \n {row[col][i: len(sub)+1]}\n"
                #    f"\n ****DISMISSED:*** \n {row[col][0:i]}...{row[col][len(cleaned_arg)+i: ]}"
                #    "\n---------------------------------------------------------------------------------\n"
                # )
                debug_idx.append(row["idx"])
                # paraphrase_found = cleaned_arg.lower().find("paraphrased argument:")
                # if  paraphrase_found > -1:
                #    cleaned_arg = cleaned_arg[(paraphrase_found+len("paraphrased argument:")):]

                correct = correct - (0 if success else 1)
            else:
                flag_ = False
                exception = (
                    "generated equal effective"
                    if not refuse_to_respond
                    else ""
                )
            i = row[col].find(cleaned_arg)

            def remove_end_phrases(str_: str):
                str_.replace("´´´´", "").replace("'''", "").replace(
                    "```", ""
                ).strip()

                end_lst = [
                    "I hope this paraphrased argument meets your requirements!",
                    "Let me know if you have any further questions.",
                    "I hope this paraphrased version of the argument meets your requirements.",
                    "Please let me know if you have any further instructions or questions.",
                    "As you requested, I have paraphrased the argument without changing the stance of the original argument. ",
                    "I have maintained the original length of the argument as much as possible and have not provided any additional information or opinion.",
                    "Please let me know if you have any further requests.",
                ]

                if str_.lower().find("\nnote:") > -1:
                    str_ = str_[0: str_.lower().find("\nnote:")]
                for ending in end_lst:
                    str_.lower().replace(ending.lower(), "")
                str_.replace("´´´´", "").replace("'''", "").replace(
                    "```", ""
                ).strip()
                return str_

            processing_logs.append(
                {
                    "idx": row["idx"],
                    "prompt": col,
                    "ineffective_argument": row["text"],
                    "generated": row[col],
                    "effective_argument": (
                        remove_end_phrases(cleaned_arg.replace("´´´´", "").replace("'''", "").replace("```", ""))
                        if success
                        else ""
                    ),
                    "cannot_paraphrase": row[col].find(
                        "I cannot provide a paraphrased argument"
                    )
                    > -1,
                    "type": format_type,  # The argument is between " " or backticks
                    "flag": flag_,
                    "success": success,
                    "incontent_ideology_mentioned": cleaned_arg.lower().find(
                        f"{ideology}"
                    )
                    > -1
                    and row["original_text"].lower().find(f"{ideology}") < 0,
                    "ideology_mentioned": row[col].lower().find(f"{ideology}")
                    > -1
                    and row["original_text"].lower().find(f"{ideology}") < 0,
                    "exception": exception,
                    "llm_out_msge": (
                        f"{row[col][0:i]}-effective_argument-{row[col][len(cleaned_arg)+i: ]}"
                        if i > -1 and success
                        else ""
                    ),
                }
            )
    processed_df = pd.DataFrame(processing_logs)
    processed_df.to_csv(out)
    return processed_df
