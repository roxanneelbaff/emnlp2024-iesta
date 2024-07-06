import ast
import datetime
import glob
import json
import time
from typing import ClassVar
import pandas as pd

from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from tqdm import tqdm
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_fixed

from langchain import HuggingFacePipeline

# --------------------------#
#           HELPERS         #
# --------------------------#

#         LLM MODELS        #
# --------------------------#


def get_llama2_model(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf", temperature: float = 0.0
):  # this is the  model used by IESTA
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=2048,
        do_sample=True,
        top_k=30,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = HuggingFacePipeline(
        pipeline=pipe, model_kwargs={"temperature": temperature}
    )
    return model


def _get_model_from_name(model_name: str, temperature: float = 0.7):
    if model_name.startswith("gpt"):
        model = ChatOpenAI(model=model_name, temperature=temperature)
    elif model_name.find("mixtra") > -1:
        model = ChatMistralAI(model=model_name, temperature=temperature)
    elif model_name.find("llama") > -1:
        model = get_llama2_model(
            "meta-llama/Llama-2-7b-chat-hf", temperature=temperature
        )  # always temperature 0 because this is how it was used
    else:
        raise ValueError(f"Invalid model name {model_name}")
    return model


#    LLM IDEOLOGY HELPER    #
# --------------------------#


def _get_ideology_role_desc(ideology_key, root: str) -> str:
    """
    Gets the idelogy description as stated in the PEW typology description
    for each ideology. The data is saved under
    *data/pew/pew_ideology_description.json*.
    """
    with open(
        f"{root}llm_ideology/pew_ideology_description.json",
        "r",
        encoding="utf8",
    ) as f:
        ideology_descs_df = pd.DataFrame(json.loads(f.read())).set_index(
            "ideology"
        )
        str_ = ideology_descs_df.loc[ideology_key]["role"]
        return str_


def _get_role_data(prompt_type, ideology, root: str = "data/"):
    with open(
        f"{root}llm_ideology/role_templates.json", "r", encoding="utf8"
    ) as f:
        role_play_prompts_df = pd.DataFrame(json.loads(f.read())).set_index(
            "name"
        )

        role_prompt_content = role_play_prompts_df.loc[prompt_type]["template"]
        role_prompt_content = (
            role_prompt_content
            if ideology != "American"
            else role_prompt_content.replace(
                "according to your ideology", "according to your role"
            )
        )
        role_prompt_content = (
            role_prompt_content
            if (ideology != "None" and ideology != "")
            else role_prompt_content.replace("according to your ideology", "")
        )
        return role_play_prompts_df, role_prompt_content


def _build_prompt_template(
    prompt_type, ideology, model_name, root: str = "data/"
):
    """
    Builds and format the final prompt for PEW question prompt
    based on four different templates, in addition to "None"
    where no role setting exists. The possible templates are
    under *role_templates/role_templates.json* and the
    pew question template is under pew/pew_prompt.txt.
    """
    if ideology != "None":
        role_play_prompts_df, role_prompt_content = _get_role_data(
            prompt_type, ideology, root
        )

        has_double = role_play_prompts_df.loc[prompt_type]["has_double_prompt"]
        prompt_template_2: str = ""
        if has_double:
            prompt_template_2 = role_play_prompts_df.loc[prompt_type][
                "prompt2"
            ][model_name]
            if ideology == "American":
                prompt_template_2 = prompt_template_2.replace(
                    "American", "American role"
                )

    # Get question-answer template
    with open(f"{root}llm_ideology/pew_prompt.txt", "r") as f:
        question_answer_prompt = f.read()  # has

    if ideology == "None":
        prompt_template = ChatPromptTemplate.from_messages(
            [("human", question_answer_prompt)],
        )
        return prompt_template

    if prompt_template_2 != "":
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", role_prompt_content),
                ("ai", prompt_template_2),
                ("human", question_answer_prompt),
            ]
        )
    else:
        prompt_template = ChatPromptTemplate.from_messages(
            [("human", f"{role_prompt_content}\n\n{question_answer_prompt}")],
        )

    return prompt_template


# --------------------------#
#         PEWQuizForLLM     #
# --------------------------#


@dataclass
class PEWQuizForLLM:
    """
    The class is reponsible for asking the 16 questions, and there follow-up questions, in the PEW quiz using an LLM.
    This class can be instantiated form an LLM model, ideology and prompt template based on role.
    """

    # follow_up_questions = [51, 111]

    model_name: str
    repeat_quiz: int = 1
    ideology_key: str = "None"
    prompt_type: str = "imagine"

    root: str = "data/"
    temperature: int = 0.7
    pew_questions: pd.DataFrame = None
    
    QUESTION_ORDER: ClassVar = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        81,
        82,
        9,
        101,
        102,
        11,
        12,
        13,
        14,
        15,
        16,
    ]

    def reload_model(self):
        self.model = None
        self.chain = None
        # Get ROLE
        self.role: str = ""

        # PEW Prompt, Initialize CHAIN
        # with open(f'{self.root}pew_prompt.txt', 'r') as f:
        self.model = _get_model_from_name(self.model_name)

        self.prompt_template = _build_prompt_template(
            self.prompt_type, self.ideology_key, self.model_name
        )
        print(self.prompt_template)
        output_parser = StrOutputParser()
        self.chain = self.prompt_template | self.model | output_parser

    def __post_init__(self):
        self.pew_questions = self.get_pew_question_answers()
        self.reload_model()

    def get_pew_question_answers(self) -> pd.DataFrame:
        pew_questions = pd.read_json(
            f"{self.root}llm_ideology/pew_question_answer.json"
        )
        pew_questions.set_index("id", inplace=True)
        return pew_questions

    def get_prompt_vals(
        self, id: int, question: str, answer_lst: list
    ) -> dict:
        answers_str = "- " + "\n- ".join(answer_lst)
        _ideology_desc = _get_ideology_role_desc(self.ideology_key, self.root)
        if self.prompt_type != "imagine":
            _ideology_desc = f"From now on {_ideology_desc}"
        prompt_attrs = {
            "ideology": (
                ""
                if (self.ideology_key == "American")
                else f" with a {self.ideology_key} ideology"
            ),
            "ideology_description": _ideology_desc,
            "id": id,
            "question": question,
            "answer_lst": answers_str,
        }
        return prompt_attrs

    def _print_model_prompt(self):
        vals_dict = self.get_prompt_vals(
            "1", "QUESTION", ["answer 1", "answer 2"]
        )
        prompt_str: str = self.prompt_template.invoke(vals_dict).to_string()
        print(prompt_str)

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(1))#, before=before_log(logger, logging.INFO))
    def prompt_llm(self, row):
        question = row["question"]
        answer_lst = ast.literal_eval(str(row["answer_lst"]))

        # BUILD PROMPT
        vals_dict = self.get_prompt_vals(row.name, question, answer_lst)
        prompt_str: str = self.prompt_template.invoke(vals_dict).to_string()

        # PROMPT LLM
        answer = self.chain.invoke(vals_dict)
        print(f"{answer} {type(answer)}")
        llm_answer: dict = (
            answer
            if isinstance(answer, dict)
            else ast.literal_eval(str(answer))
        )
        llm_answer = {str(k): v for k, v in llm_answer.items()}
        return prompt_str, llm_answer

    def take_pew_quiz(self) -> pd.DataFrame:
        llm_answer_arr = []

        def ask_pew_format_answer(row_, is_follow_up: bool = False):
            record = {}
            question_id: int = row_.name
            prompt, llm_answer = self.prompt_llm(row_)
            if str(question_id) not in llm_answer.keys():
                print(f"question_id: {question_id}, {str(llm_answer)}")
            if len(llm_answer) != 1:
                print(f"question_id: {question_id}, {str(llm_answer)}")

            record["id"] = question_id
            record["iteration"] = self.iteration
            record["prompt"] = prompt
            record["question"] = row_["question"]
            record["answer_lst"] = row_["answer_lst"]
            record["is_follow_up"] = is_follow_up
            record["llm_output"] = llm_answer
            record["answer"] = llm_answer[str(question_id)]
            return record

        for question_id in PEWQuizForLLM.QUESTION_ORDER:
            row = self.pew_questions.loc[question_id]

            llm_record = ask_pew_format_answer(row)
            llm_answer_arr.append(llm_record)

            if row["has_follow_up"]:
                answer_followup_id_dict = ast.literal_eval(
                    str(row["follow_up_conditions"])
                )
                answer_str = llm_record["answer"]
                if answer_str in answer_followup_id_dict.keys():
                    followup_question_id = answer_followup_id_dict[answer_str]
                    followup_row = self.pew_questions.loc[followup_question_id]
                    followup_llm_record = ask_pew_format_answer(
                        followup_row, is_follow_up=True
                    )
                    llm_answer_arr.append(followup_llm_record)

        return pd.DataFrame(llm_answer_arr)

    def repeat_pew_quiz(self):
        all_iterations: list[pd.DataFrame] = []

        def _get_file_names(model_name):
            temp: str = f"T{self.temperature}"
            prompt_type_str = self.prompt_type.upper()
            desc: str = (
                f"prompt-{prompt_type_str}_ideology-{self.ideology_key.replace(' ', '').upper()}"
            )
            return f"{self.root}llm_ideology/pew_quiz_results/pew_{model_name}_{desc}_{temp}_i"

        fname = _get_file_names(self.model_name)

        num_existing_iterations = len(glob.glob(f"{fname}*.csv"))
        if num_existing_iterations >= self.repeat_quiz:
            print("Iterations are done...")
            return
        for iteration in range(
            (num_existing_iterations + 1), self.repeat_quiz + 1
        ):
            u = int(datetime.datetime.now().timestamp() * 1000)

            self.reload_model()
            self.iteration = iteration
            print(f"ITERATION {iteration}")
            quiz_df: pd.DataFrame = self.take_pew_quiz()

            quiz_df.to_csv(
                f"{fname}{iteration}_stamp{u}.csv",
                index=False,
            )
            all_iterations.append(quiz_df)
            time.sleep(2)
        return pd.concat(all_iterations, ignore_index=True)


# SIDE EXPERIMENTS
# PROMPT 2 generation for role4


def get_prompt2_samples(
    model_lst: list,
    ideology_lst: list,
    temperature: float = 0.7,
    repetitions: int = 50,
    root: str = "data/",
):
    prompt1_results = []
    for iteration in range(0, repetitions):
        for model_name in tqdm(model_lst):
            # Chain
            temperature = 0.7
            if model_name.startswith("gpt"):
                model = ChatOpenAI(model=model_name, temperature=temperature)
            elif model_name.find("mistra") > -1:
                model = ChatMistralAI(
                    model=model_name, temperature=temperature
                )

            for ideology in ideology_lst:
                record = {
                    "model": model_name,
                    "ideology": ideology,
                    "iteration": iteration,
                    "id": f"{model_name}_{ideology}",
                }

                _, role_prompt_content = _get_role_data(
                    "role_prompt_4", ideology, root
                )

                prompt_template = ChatPromptTemplate.from_messages(
                    [("human", role_prompt_content)],
                )

                _ideology_desc = _get_ideology_role_desc(ideology, root)

                # PROMPT ATTR
                prompt_attrs = {
                    "ideology": (
                        ""
                        if (ideology == "American")
                        else f" with a {ideology} ideology"
                    ),
                    "ideology_description": _ideology_desc,
                }
                record["prompt"] = prompt_template.invoke(
                    prompt_attrs
                ).to_string()

            output_parser = StrOutputParser()
            chain = prompt_template | model | output_parser
            record["response"] = chain.invoke(prompt_attrs)

            prompt1_results.append(record)

    prompt2_df = pd.DataFrame(prompt1_results)
    prompt2_df.to_csv(
        f"{root}ll_personality/experiment_prompt2_generation.csv"
    )
    return prompt2_df


def get_top_prompt2(root):
    """
    Interprets results from get_prompt2_samples
    """
    prompt1_df = pd.read_csv(
        f"{root}llm_ideology/experiment_prompt2_generation.csv"
    )

    def _apply(row):
        row["ideology"] = (
            row["ideology"][:-1]
            if row["ideology"].endswith("s")
            else row["ideology"]
        )
        return row

    prompt1_df = prompt1_df.apply(_apply, axis=1)

    prompt1_majority_per_model = []
    for model_ideo, df_ in prompt1_df.groupby(["id"]):
        dict_ = {"model_ideo": model_ideo}
        # for _, row_ in tqdm(df_.iterrows()):
        # print(f"{row_['iteration']}: {row_['response']}")
        df_ = df_[
            ~df_["response"].str.startswith("As an AI, I don't have personal")
        ]
        df_ = df_[
            ~df_["response"].str.startswith("As an artificial intelligence")
        ]
        df_ = df_[~df_["response"].str.startswith("As an AI developed")]
        if len(df_) == 0:
            print(model_ideo)
            continue
        dict_["response"] = (
            df_["response"].value_counts().to_frame().iloc[0].name
        )

        dict_["count"] = (
            df_["response"].value_counts().to_frame().iloc[0]["response"]
        )

        print("--> MODEL ", model_ideo)
        print(dict_["response"])

        prompt1_majority_per_model.append(dict_)
        # print(dict_)
        # print()
    prompt1_maj_df = pd.DataFrame(prompt1_majority_per_model)
    for _, row in prompt1_maj_df.iterrows():
        print(row["model_ideo"])
        print(row["response"])
    return prompt1_maj_df
