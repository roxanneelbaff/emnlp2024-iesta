from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import os

import pandas as pd
from tqdm.notebook import tqdm

from langchain.chat_models import ChatOpenAI
import transformers
from dotenv import load_dotenv, find_dotenv

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import torch

from datasets import load_dataset, Dataset
from iesta.llms import prompts

from iesta.machine_learning.huggingface_loader import IESTAHuggingFace
from ydata_profiling import ProfileReport
import json
import pandas as pd
from tqdm import tqdm
from os.path import exists
import dataclasses

from datasets import load_dataset, Dataset
from datasets.combine import concatenate_datasets
from ydata_profiling import ProfileReport
from langdetect import detect
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import ClassVar
from langchain.llms import HuggingFaceTextGenInference
from iesta.llms.models import IestaLLM
import datasets

@dataclasses.dataclass
class Generator:
    ideology: str  # liberal or conservative
    llm_model: IestaLLM  # gpt or alpaca llama-v2-70b-chat
    root_path: str = ""

    data_limit: int = 500
    
    seed: int = 2062021

    out_file: str = "data/llms_out/new/"

    n_shots: int = 0  # we use 1 to 3
    flag_profile_training_data: bool = True
    flag_profile_test_data: bool = False
    data_save: bool = False
    _temp_flag: bool = True



    # ---- End Helpers --- #

    def __post_init__(self):
        print("********VERSION********")
        found = load_dotenv(f"{self.root_path}/.env")
        print(f"dotenv was found: {found}")

        print("Initializing all prompt templates in variable prompt_dict..")
        self.prompt_dict = prompts.get_all_instructions_per_ideology(self.ideology)

        print(
            f"Initializing LLM for {self.model_name} in variable local_llm..."
        )
        
        print(
            f"Getting filtered dataset top {self.data_limit} in variable"
            "filtered_dataset..."
        )
        self.filtered_dataset = self.get_data(effect="ineffective")

        if self.n_shots > 0:
            self.examples = self.get_examples()


    def get_data(self, effect="ineffective"):
        limit = Generator._LIMIT_
        seed = 2062021
        name: str = f"notaphoenix/debateorg_w_effect_for_{self.ideology}"
        dataset: Dataset = load_dataset(name, split="test")
        dataset = dataset.filter(
            lambda x: x["label"] == IESTAHuggingFace._LABEL2ID_[effect]
        ).shuffle(seed=seed)

        if len(dataset) > limit:
            dataset = dataset.select(range(limit))

        print(f"{len(dataset)} before len filter")
        dataset = dataset.filter(
            lambda x: len(x["text"].split(" ")) > 10
            and len(x["text"].split(" ")) <= 1024
            and x["idx"] != 64707
            and detect(x["text"]) == "en"
        )
        print(f"{len(dataset)} after len filter")

        while len(dataset) < limit:
            idxes = dataset.to_pandas()["idx"].values.tolist()
            dataset_extra: Dataset = load_dataset(name, split="test")
            dataset_extra = dataset_extra.filter(
                lambda x: len(x["text"].split(" ")) > 10
                and len(x["text"].split(" ")) <= 1024
                and x["idx"] != 64707
                and detect(x["text"]) == "en"
            )

            dataset_extra = dataset_extra.filter(
                lambda x: x["label"] == IESTAHuggingFace._LABEL2ID_[effect]
                and ["idx"] not in idxes
            ).shuffle(seed=seed)
            dataset_extra = dataset_extra.select(range(limit - len(dataset)))
            print(f"{len(dataset_extra)} of extra")
            dataset = concatenate_datasets([dataset, dataset_extra])

            print(f"{len(dataset)} new length")
        print(f"Return dataset {name} with {len(dataset)} ")
        # dataset = dataset.map(lambda example, idx: {"id": idx, **example}, with_indices=True)

        df = dataset.to_pandas().copy()
        if self.data_profiling:
            report = ProfileReport(df=df, minimal=False)
            report.to_file(f"{self.ideology}_test_{limit}_seed_{seed}")

        if self.data_save:
            df.to_csv(f"data/{self.ideology}_test_{limit}_seed_{seed}.csv")
        return dataset

    def _run_test(self, instructions: str = "", 
                  input_prompt: str = "\nArgument: ",
                  chatPrompt: bool = True):
        if chatPrompt:
            system_message_prompt = SystemMessagePromptTemplate.from_template(
             instructions+self.prompt_dict["all"].format(ideology=self.ideology)
            )
            human_template = input_prompt+"{ineffective_argument}"
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                human_template
            )
            chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt]
            )
        else:
            template = (
                self.prompt_dict["all"].format(ideology=self.ideology) + input_prompt +"{ineffective_argument}\n"
                )

            chat_prompt = PromptTemplate(
                template=template,
                input_variables=["ineffective_argument"],
            )

        llm_chain = LLMChain(llm=self.llm_model.llm, prompt=chat_prompt)
        result = llm_chain.run(
            ineffective_argument="Climate change "
            "litigations are now linked to human rights. "
            "This does not make sense because climate "
            "change is not caused by humans and we should "
            "protect the rights of our fathers"
        )

        return result

    def get_examples(self) -> list:
        examples_per_category = {}
        if self.n_shots == 0:
            return
        
        limit = Generator._LIMIT_ * self.n_shots
        seed = self.seed

        name: str = f"notaphoenix/debateorg_w_effect_for_{self.ideology}"
        dataset: Dataset = load_dataset(name, split="training")
        dataset = dataset.filter(
            lambda x: x["label"] == IESTAHuggingFace._LABEL2ID_["effective"]
            and len(x["text"].split(" ")) > 10
            and len(x["text"].split(" ")) <= 1024
            and detect(x["text"]) == "en"
        ).shuffle(seed=seed)

        category_counts = self.filtered_dataset.to_pandas()['category'].value_counts().to_frame('counts')
        for category, count_row in category_counts.iterrows():
            count = count_row['counts']
            _dataset = dataset.filter(
                lambda x: x["category"] == category
            )
            category_examples = _dataset
            if len(category_examples) > count:
                category_examples = category_examples.select(range(count))
            
            while len(category_examples) < count:
                reselect_num = min(len(_dataset), (count-len(category_examples)))
                
                category_examples = datasets.concatenate(category_examples,
                                                         category_examples.select(range(reselect_num))
                                                         )

            print(f"{len(category_examples)}  length for category {category} with test # {count}")

            df = category_examples.to_pandas().copy()
            filename: str = f"{self.ideology}_training_{limit}_seed{seed}" \
                            f"_{self.n_shots}shot_{category}"
            if self.trainingdata_profiling:
                report = ProfileReport(df=df, minimal=True)
                report.to_file(f"data/llms_out/fewshot_examples/{filename}.html")

            df.to_csv(f"data/llms_out/fewshot_examples/{filename}.csv")
            examples_per_category[category] =  [
                {"effective_argument": x} for x in df["text"].values.tolist()
            ]
            
        return examples_per_category

    # GENERATION #
    # ---------- #
    def generate_for_prompts(self, ineffective_argument: str, category: str= None):
        result_dict = {}
        print("generate_for_prompts called")
        # Preparing PROMPTS

        assert (self.n_shots > 0 and category is not None) or self.n_shots == 0

        local_examples = local_examples = [
            self.examples[category].pop()
            for _ in range(0, self.n_shots)
        ] if self.n_shots > 0 and self.examples is not None and len(self.examples[category]) > 0 


        for k, instructions in self.prompt_dict.items():
            if self.n_shots > 0:

                example_prompt = PromptTemplate(
                    input_variables=["effective_argument"],
                    template="This is an example of an effective argument: {effective_argument}",
                )

                prompt = FewShotPromptTemplate(
                    examples=local_examples,
                    example_prompt=example_prompt,
                    suffix=self.llm_model.get_prompt_template(instructions.format(ideology=self.ideology)),
                    input_variables=["text"],
                )
            else:  # 0 shot
                prompt = self.llm_model.get_prompt_template(instructions.format(ideology=self.ideology))

            # remove
            if self._temp_flag:
                print("****** prompt: ")
                print(prompt.format(text=ineffective_argument))
                self._temp_flag = False

            # End of remove
            llm_chain = LLMChain(llm=self.llm_model.llm, prompt=prompt)
            result_dict[k] = llm_chain.run(
                text=ineffective_argument
            )
            result_dict[f"len_{k}"] = len(result_dict[k])
            result_dict["len_orig"] = len(ineffective_argument)
            if self.n_shots > 0:
                result_dict["examples"] = local_examples

        return result_dict

    def generate_all(self):
        fewshots_text = (
            f"_{self.n_shots}shots"
            if self.n_shots > 0
            else ""
        )

        out_file = f"{self.out_file}{self.ideology}_{self.model_name.lower()}{fewshots_text}.jsonl"

        existing_indices = []
        if exists(out_file):
            _df = pd.read_json(path_or_buf=out_file, lines=True)
            existing_indices = _df["idx"].values.tolist()

        add_new_l = False
        if len(existing_indices) > 0:
            print(f"filtering out existing indices ({len(existing_indices)})")
            self.filtered_dataset = self.filtered_dataset.filter(
                lambda example: example["idx"] not in existing_indices
            )
            print(f"{self.filtered_dataset.num_rows} to go...")
            add_new_l = True

        with open(out_file, "a") as file:
            for datapoint in tqdm(self.filtered_dataset):
                try:
                    promt_generated_dict = self.generate_for_prompts(
                        datapoint["text"], datapoint["category"]
                    )
                    promt_generated_dict.update(datapoint)
                    nline = "\n" if add_new_l else ""

                    file.write(f"{nline}{json.dumps(promt_generated_dict)}")
                    add_new_l = True
                except Exception as e:
                    print(e)
                    print(
                        f"Failed to get a response for ID: {datapoint['idx']}"
                    )
