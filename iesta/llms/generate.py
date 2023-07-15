from langchain.llms import OpenAIChat
from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter,TokenTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.few_shot import FewShotPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
pd.set_option('display.max_colwidth', None)
from tqdm.notebook import tqdm
tqdm.pandas()
from ast import literal_eval
import random
import re
from langchain.chat_models import ChatOpenAI
from torch import cuda, bfloat16 
import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM 
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv, find_dotenv
import torch

from datasets import load_dataset, Dataset

from iesta.machine_learning.huggingface_loader import IESTAHuggingFace
from ydata_profiling import ProfileReport
import json
import pandas as pd
from tqdm import tqdm
from os.path import exists
import dataclasses

from datasets import load_dataset, Dataset
from datasets.combine import concatenate_datasets
from iesta.machine_learning.huggingface_loader import IESTAHuggingFace
from ydata_profiling import ProfileReport
from langdetect import detect

@dataclasses.dataclass
class Generator():
    _MODEL_CHATGPT_ = "gpt-3.5-turbo"
    _MODEL_ALPACA_ = "alpaca"
    ideology: str  # liberal or conservative
    model_name: str  # gpt or alpaca

    data_limit: int = 500
    data_profiling: bool = False
    data_save: bool = False
    seed: int = 2062021

    out_file: str = "data/llms_out/"

    # HELPERS   #
    # --------- #
    @staticmethod
    def init_prompts():
        basic_str = "Transform the following argument to an effective argument by maintaining the original length"
        ideology_str = "for readers with a {ideology} political ideology"
        content_str = "by preserving the content of the argument"
        style_str = "by only changing the style of the text"

        prompt_dict = {
            "basic": f"{basic_str}:",
            "ideology": f"{basic_str} {ideology_str}:",
            "content": f"{basic_str} {content_str}:",
            "style": f"{basic_str} {style_str}:",
            "ideology-content": f"{basic_str} {ideology_str} {content_str}:",
            "ideology-style": f"{basic_str} {ideology_str} {style_str}:",
            "all": f"{basic_str} {ideology_str} {content_str} and {style_str}:",
        }
        return prompt_dict

    @staticmethod
    def create_prompt_template(prompt):
        system_message_prompt = SystemMessagePromptTemplate.from_template(prompt)
        human_template = "Argument: {ineffective_argument}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template
            )

        return [system_message_prompt, human_message_prompt]

    # ---- End Helpers --- #

    def __post_init__(self):
        found = load_dotenv(find_dotenv())
        print(f"dotenv was found: {found}")

        print("Initializing all prompt templates in variable prompt_dict..")
        self.prompt_dict = Generator.init_prompts()

        print(f"Initializing LLM for {self.model_name} in variable local_llm...")
        self.local_llm = self.get_model()

        print(f"Getting filtered dataset top {self.data_limit} in variable"
              "filtered_dataset...")
        self.filtered_dataset = self.get_data(effect="ineffective")

    def get_model(self):
        if self.model_name == Generator._MODEL_CHATGPT_:
            local_llm = ChatOpenAI(model_name=Generator._MODEL_CHATGPT_,
                                   temperature=0)
        elif self.model_name == Generator._MODEL_ALPACA_:
            # device = torch.device('cuda:0')
            ALPACA_WEIGHTS_FOLDER = "/localdata1/EmEx/model_weights/alpaca_7b"

            tokenizer = AutoTokenizer.from_pretrained(ALPACA_WEIGHTS_FOLDER)
            alpaca_llm = AutoModelForCausalLM.from_pretrained(
                ALPACA_WEIGHTS_FOLDER,
                #load_in_8bit=True,
                device_map='cuda:1',
            )

            pipe = pipeline(
                "text-generation",
                model=alpaca_llm,
                tokenizer=tokenizer,
                max_length=2048,
                temperature=0,
                top_p=0.95,
                repetition_penalty=1.2
            )
            local_llm = HuggingFacePipeline(pipeline=pipe)
        return local_llm

    def get_data(self, effect='ineffective', limit: int = 500):
        seed = 2062021
        name: str = f'notaphoenix/debateorg_w_effect_for_{self.ideology}'
        dataset: Dataset = load_dataset(name, split="test")
        dataset = dataset.filter(lambda x: x["label"] == IESTAHuggingFace._LABEL2ID_[effect]).shuffle(seed=seed)

        if len(dataset) > limit:
            dataset = dataset.select(range(limit))

        print(f"{len(dataset)} before len filter")
        dataset = dataset.filter(lambda x: len(x["text"].split(" ")) > 10 and \
                                           len(x["text"].split(" ")) <= 1024 and  \
                                           x["idx"] != 64707 and  \
                                           detect(x["text"]) == "en")
        print(f"{len(dataset)} after len filter")

        while len(dataset) < limit:
            idxes = dataset.to_pandas()['idx'].values.tolist()
            dataset_extra: Dataset = load_dataset(name, split="test")
            dataset_extra = dataset_extra.filter(lambda x: len(x["text"].split(" ")) > 10 and \
                                                           len(x["text"].split(" ")) <= 1024 and  \
                                                           x["idx"] != 64707 and  \
                                                           detect(x["text"]) == "en")

            dataset_extra = dataset_extra.filter(lambda x:  x["label"] == IESTAHuggingFace._LABEL2ID_[effect]  and \
                                                            x["idx"] not in idxes).shuffle(seed=seed)
            dataset_extra = dataset_extra.select(range(limit-len(dataset)))
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
            df.to_csv(f"{self.ideology}_test_{limit}_seed_{seed}.csv")
        return dataset

    def _run_test(self):
        chat_prompt = ChatPromptTemplate.from_messages(
            Generator.create_prompt_template(self.prompt_dict["all"].format(
                                             ideology=self.ideology)
                                             ))

        llm_chain = LLMChain(llm=self.local_llm, prompt=chat_prompt)
        result = llm_chain.run(ineffective_argument="Climate change "
                                                    "litigations are now linked to human rights. ")

        return result

    def generate_for_prompts(self, ineffective_argument: str):

        result_dict = {}
        if self.model_name == Generator._MODEL_CHATGPT_:
            for k, prompt_template in self.prompt_dict.items():
                chat_prompt = ChatPromptTemplate.from_messages(
                    Generator.create_prompt_template(prompt_template.format(ideology=self.ideology)))
                llm_chain = LLMChain(llm=self.local_llm, prompt=chat_prompt)
                result_dict[k] = llm_chain.run(ineffective_argument=ineffective_argument)
                result_dict[f"len_{k}"] = len(result_dict[k])
                result_dict["len_orig"] = len(ineffective_argument)
        else:
            for k, prompt_template in self.prompt_dict.items():
                template = prompt_template.format(ideology=self.ideology) +"\n    Argument: {ineffective_argument}\n    "
                prompt = PromptTemplate(template=template, input_variables=["ineffective_argument"])

                llm_chain = LLMChain(llm=self.local_llm, prompt=prompt)
                result_dict[k] = llm_chain.run(ineffective_argument)
                result_dict[f"len_{k}"] = len(result_dict[k])
                result_dict["len_orig"] = len(ineffective_argument)

        return result_dict

    def generate_all(self):
        out_file = f"{self.out_file}{self.ideology}_{self.model_name.lower()}_2.jsonl"

        existing_indices = []
        if exists(out_file):
            _df = pd.read_json(path_or_buf=out_file, lines=True)
            existing_indices = _df['idx'].values.tolist()

        add_new_l = False
        if len(existing_indices) > 0 :
            print(f"filtering out existing indices ({len(existing_indices)})")
            self.filtered_dataset = self.filtered_dataset.filter(lambda example: example['idx'] not in existing_indices)
            print(f"{self.filtered_dataset.num_rows} to go...")
            add_new_l = True

        with open(out_file, 'a') as file:
            for datapoint in tqdm(self.filtered_dataset):
                try:
                    promt_generated_dict = self.generate_for_prompts(datapoint["text"])
                    promt_generated_dict.update(datapoint)
                    nline = "\n" if add_new_l else ""

                    file.write(f"{nline}{json.dumps(promt_generated_dict)}")
                    add_new_l = True
                except Exception as e:
                    print(e)
                    print(f"Failed to get a response for ID: {datapoint['idx']}")   
