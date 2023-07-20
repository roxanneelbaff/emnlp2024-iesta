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


@dataclasses.dataclass
class Generator:
    ideology: str  # liberal or conservative
    model_name: str  # gpt or alpaca llama-v2-70b-chat
    root_path: str = ""

    data_limit: int = 500
    data_profiling: bool = False
    data_save: bool = False
    seed: int = 2062021

    out_file: str = "data/llms_out/"

    use_fewshots: bool = False
    fewshots_num_examples: int = 1  # we use 1 to 3
    fewshots_w_semantic_similarity: bool = False
    verbose: int = 0
    trainingdata_profiling: bool = True
    _temp_flag: bool = True

    _MODEL_CHATGPT_: ClassVar = "gpt-3.5-turbo"
    _MODEL_ALPACA_: ClassVar = "alpaca"
    _MODEL_LLAMA_V2_70B_: ClassVar = "llama-v2-70b-chat"
    _MODEL_LLAMA_V2_7B_: ClassVar = "llama-v2-7b"
    _MODEL_FALCON_INST_40B_: ClassVar = "falcon-40b"

    RUNPOD_ID_DIC: ClassVar ={
        _MODEL_LLAMA_V2_70B_: "nle29nkbxmmvnf",
        _MODEL_FALCON_INST_40B_: "kab3zkyrnrugd1",
        _MODEL_LLAMA_V2_7B_: "sx1co9yds7i7je"
    }
    _LIMIT_: ClassVar = 500

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
            "content": f"{basic_str}, and {content_str}:",
            "style": f"{basic_str}, and {style_str}:",
            "ideology-content": f"{basic_str} {ideology_str} {content_str}:",
            "ideology-style": f"{basic_str} {ideology_str} {style_str}:",
            "all": f"{basic_str} {ideology_str} {content_str}, and {style_str}:",
        }
        return prompt_dict

    @staticmethod
    def create_prompt_template(prompt):
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            prompt
        )
        human_template = "Argument: {ineffective_argument}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template
        )

        return [system_message_prompt, human_message_prompt]

    # ---- End Helpers --- #

    def __post_init__(self):
        print("********VERSION 64********")
        found = load_dotenv(f"{self.root_path}/.env")
        print(f"dotenv was found: {found}")

        print("Initializing all prompt templates in variable prompt_dict..")
        self.prompt_dict = Generator.init_prompts()

        print(
            f"Initializing LLM for {self.model_name} in variable local_llm..."
        )
        self.local_llm = self.get_model()

        print(
            f"Getting filtered dataset top {self.data_limit} in variable"
            "filtered_dataset..."
        )
        self.filtered_dataset = self.get_data(effect="ineffective")

        if self.use_fewshots:
            self.examples = self.get_examples()

    def get_model(self):
        if self.model_name == Generator._MODEL_CHATGPT_:
            local_llm = ChatOpenAI(
                model_name=Generator._MODEL_CHATGPT_, temperature=0
            )
        elif self.model_name == Generator._MODEL_ALPACA_:
            # device = torch.device('cuda:0')
            ALPACA_WEIGHTS_FOLDER = "/localdata1/EmEx/model_weights/alpaca_7b"

            tokenizer = AutoTokenizer.from_pretrained(ALPACA_WEIGHTS_FOLDER)
            alpaca_llm = AutoModelForCausalLM.from_pretrained(
                ALPACA_WEIGHTS_FOLDER,
                # load_in_8bit=True,
                device_map="cuda:1",
            )

            pipe = pipeline(
                "text-generation",
                model=alpaca_llm,
                tokenizer=tokenizer,
                max_length=4096,
                temperature=0,
                top_p=0.95,
                repetition_penalty=1.2,
            )
            local_llm = HuggingFacePipeline(pipeline=pipe)
        else:  # if self.model_name == Generator._MODEL_LLAMA_V2_70B_:
            pod_id = Generator.RUNPOD_ID_DIC[self.model_name]
            print(pod_id)
            # pod_id = "nle29nkbxmmvnf"
            # print(pod_id)

            inference_server_url = f"https://{pod_id}-80.proxy.runpod.net"
            local_llm = HuggingFaceTextGenInference(
                inference_server_url=inference_server_url,
                max_new_tokens=1024,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
            )
        print(type(local_llm))
        return local_llm

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

        llm_chain = LLMChain(llm=self.local_llm, prompt=chat_prompt)
        result = llm_chain.run(
            ineffective_argument="Climate change "
            "litigations are now linked to human rights. "
            "This does not make sense because climate "
            "change is not caused by humans and we should "
            "protect the rights of our fathers"
        )

        return result

    def get_examples(self, save: bool = True) -> list:
        limit = Generator._LIMIT_ * self.fewshots_num_examples
        seed = self.seed

        name: str = f"notaphoenix/debateorg_w_effect_for_{self.ideology}"
        dataset: Dataset = load_dataset(name, split="training")
        dataset = dataset.filter(
            lambda x: x["label"] == IESTAHuggingFace._LABEL2ID_["effective"]
            and len(x["text"].split(" ")) > 10
            and len(x["text"].split(" ")) <= 1024
            and detect(x["text"]) == "en"
        ).shuffle(seed=seed)

        if len(dataset) > limit and not self.fewshots_w_semantic_similarity:
            dataset = dataset.select(range(limit))
        print(f"{len(dataset)} new length")
        print(f"Return dataset {name} with {len(dataset)} ")

        df = dataset.to_pandas().copy()
        filename: str = f"{self.ideology}_training_{limit}_seed_{seed}_fewshot_{self.fewshots_num_examples}_similarity{self.fewshots_w_semantic_similarity}"
        if self.trainingdata_profiling:
            report = ProfileReport(df=df, minimal=True)
            report.to_file(f"data/{filename}.html")

        df.to_csv(f"data/{filename}.csv")

        result = [
            {"effective_argument": x} for x in df["text"].values.tolist()
        ]
        print(result[:3])
        return result

    # GENERATION #
    # ---------- #
    def generate_for_prompts(self, ineffective_argument: str):
        result_dict = {}
        print("generate_for_prompts called")
        # Preparing PROMPTS
        local_examples = []
        if self.use_fewshots:
            if len(self.examples) < self.fewshots_num_examples:
                print("warning: ran out of examples, replenishing...")
                self.examples = self.get_examples(save=False)

            local_examples = [
                self.examples.pop()
                for _ in range(0, self.fewshots_num_examples)
            ]

        for k, prompt_template in self.prompt_dict.items():
            if self.use_fewshots:
                template = (
                    prompt_template.format(ideology=self.ideology)
                    + "\n    Argument: {ineffective_argument}\n"
                )
                # prompt = PromptTemplate(template=template, input_variables=["ineffective_argument"])

                example_prompt = PromptTemplate(
                    input_variables=["effective_argument"],
                    template="Example of an argument that has an effective style: {effective_argument}",
                )

                prompt = FewShotPromptTemplate(
                    examples=local_examples,
                    example_prompt=example_prompt,
                    suffix=template,
                    input_variables=["ineffective_argument"],
                )
            else:  # 0 shot
                if self.model_name == Generator._MODEL_CHATGPT_ or self.model_name.startswith("llama-v2") or self.model_name.startswith("falcon"):
                    prompt = ChatPromptTemplate.from_messages(
                        Generator.create_prompt_template(
                            prompt_template.format(ideology=self.ideology)
                        )
                    )
                else:
                    template = (
                        prompt_template.format(ideology=self.ideology)
                        + "\n    Argument: {ineffective_argument}\n    "
                    )
                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["ineffective_argument"],
                    )

            # remove
            if self._temp_flag:
                print("****** prompt: ")
                print(prompt.format(ineffective_argument=ineffective_argument))
                self._temp_flag = False

            # End of remove
            llm_chain = LLMChain(llm=self.local_llm, prompt=prompt)
            result_dict[k] = llm_chain.run(
                ineffective_argument=ineffective_argument
            )
            result_dict[f"len_{k}"] = len(result_dict[k])
            result_dict["len_orig"] = len(ineffective_argument)
            if self.use_fewshots:
                result_dict["examples"] = local_examples

        return result_dict

    def generate_all(self):
        fewshots_text = (
            f"_{self.fewshots_num_examples}fewshots"
            if self.use_fewshots
            else ""
        )
        fewshots_text = (
            f"{fewshots_text}_with_similarity"
            if self.fewshots_w_semantic_similarity
            else fewshots_text
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
                        datapoint["text"]
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
