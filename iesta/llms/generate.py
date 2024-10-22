

import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv

from langchain import PromptTemplate, LLMChain


from datasets import load_dataset, Dataset
from iesta.llms import prompts

from iesta.data.huggingface_loader import IESTAHuggingFace
import json

from os.path import exists
import dataclasses

from datasets.combine import concatenate_datasets
from langdetect import detect
from langchain.prompts.few_shot import FewShotPromptTemplate

from iesta.llms.models import IestaLLM
import datasets
from pathlib import Path


@dataclasses.dataclass
class Generator:
    ideology: str  # liberal or conservative
    llm_model: IestaLLM  # gpt or alpaca llama-v2-70b-chat
    root_path: str = None
    n_shots: int = 0  # we use 1 to 3
    flag_profile_training_data: bool = True
    flag_profile_test_data: bool = False
    data_save: bool = True
    data_limit: int = 500
    out_file: str = "llms_out/new/"
    out_ext: str = ""  # add steered
    seed: int = 2062021
    _temp_flag: bool = True

    # ---- End Helpers --- #

    def __post_init__(self):
        print("********VERSION********")
        if self.root_path is None:
            self.root_path = "../data/"
        self.out_file = f"{self.root_path}{self.out_file}"

        found = load_dotenv(f"{self.root_path}../.env")
        print(f"dotenv was found: {found}")

        print("Initializing all prompt templates in variable prompt_dict..")
        self.prompt_dict = prompts.get_all_instructions_per_ideology(
            self.ideology
        )

        print(
            f"Getting filtered dataset top {self.data_limit} in variable"
            "filtered_dataset..."
        )
        self.filtered_dataset = self.get_data(effect="ineffective")

        if self.n_shots > 0:
            self.examples = self.get_examples()

    def get_data(self, effect="ineffective"):
        out_path = f"{self.root_path}out/{self.ideology}_idx.csv"
        preset_indices = []
        print(f"{out_path}")
        if Path(out_path).is_file():
            print("original file found")
            preset_indices = pd.read_csv(out_path)["idx"].values.tolist()[:500]
        
        seed = 2062021
        name: str = f"notaphoenix/debateorg_w_effect_for_{self.ideology}"
        dataset: Dataset = load_dataset(name, split="test")

        dataset = dataset.filter(
            lambda x: x["label"] == IESTAHuggingFace._LABEL2ID_[effect]
        ).shuffle(seed=seed)

        if len(preset_indices) > 0:
            print(f"filtering using Indices with len {len(preset_indices)}")
            dataset = dataset.filter(
                lambda x: x["idx"] in preset_indices
                )
            print(f"using preset indices {len(dataset)}")
            return dataset

        if len(dataset) > self.data_limit:
            dataset = dataset.select(range(self.data_limit))

        print(f"{len(dataset)} before len filter")
        dataset = dataset.filter(
            lambda x: len(x["text"].split(" ")) > 10
            and len(x["text"].split(" ")) <= 1024
            and x["idx"] != 64707
            and detect(x["text"]) == "en"
        )
        print(f"{len(dataset)} after len filter")

        while len(dataset) < self.data_limit:
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
            dataset_extra = dataset_extra.select(
                range(self.data_limit - len(dataset))
            )
            print(f"{len(dataset_extra)} of extra")
            dataset = concatenate_datasets([dataset, dataset_extra])

            print(f"{len(dataset)} new length")
        print(f"Return dataset {name} with {len(dataset)} ")
        # dataset = dataset.map(lambda example, idx: {"id": idx, **example}, with_indices=True)

        df = dataset.to_pandas().copy()
        #if self.flag_profile_test_data:
        #    report = ProfileReport(df=df, minimal=False)
        #    report.to_file(
        #        f"{self.root_path}llms_out/data_profile_{self.ideology}_test{self.data_limit}_seed{seed}.html"
        #    )

        if self.data_save:
            df.to_csv(
                f"{self.root_path}llms_out/data_{self.ideology}_test{self.data_limit}_seed{seed}.csv"
            )
        return dataset

    def _run_test(self):
        instructions = self.prompt_dict["all"]

        llm_chain = LLMChain(
            llm=self.llm_model.llm,
            prompt=self.llm_model.get_prompt_template(instructions),
        )
        result = llm_chain.run(
            test="Climate change "
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

        limit = self.data_limit * self.n_shots
        seed = self.seed

        name: str = f"notaphoenix/debateorg_w_effect_for_{self.ideology}"
        dataset: Dataset = load_dataset(name, split="training")
        dataset = dataset.filter(
            lambda x: x["label"] == IESTAHuggingFace._LABEL2ID_["effective"]
            and len(x["text"].split(" ")) > 10
            and len(x["text"].split(" ")) <= 1024
            and detect(x["text"]) == "en"
        ).shuffle(seed=seed)

        category_counts = (
            self.filtered_dataset.to_pandas()["category"]
            .value_counts()
            .to_frame("counts")
        )
        for original_category, count_row in category_counts.iterrows():
            # No movies in training, get from similar category, art
            if self.ideology == "liberal":
                category = (
                    "Arts"
                    if original_category in ["Movies", "Music"]
                    else original_category
                )
            else:
                category = (
                    "Entertainment"
                    if original_category
                    in ["Fashion", "Cars", "Places-Travel"]
                    else original_category
                )

            count = count_row["counts"]
            _dataset = dataset.filter(lambda x: x["category"] == category)
            print(f"category {original_category} has {len(_dataset)}")
            category_examples = _dataset
            if len(category_examples) > count:
                category_examples = category_examples.select(range(count))

            while len(category_examples) < count:
                reselect_num = min(
                    len(_dataset), (count - len(category_examples))
                )
                print("reselecting")
                category_examples = datasets.concatenate_datasets(
                    [
                        category_examples,
                        category_examples.select(range(reselect_num)),
                    ]
                )

            print(
                f"category {original_category} done with {len(category_examples)}/{count}"
            )

            df = category_examples.to_pandas().copy()
            filename: str = (
                f"{self.ideology}_training_{limit}_seed{seed}"
                f"_{self.n_shots}shot_{original_category}"
            )
            #if self.flag_profile_training_data:
            #    report = ProfileReport(df=df, minimal=True)
            #    report.to_file(
            #        f"{self.root_path}llms_out/fewshot_examples/{filename}.html"
            #    )

            df.to_csv(
                f"{self.root_path}llms_out/fewshot_examples/{filename}.csv"
            )
            examples_per_category[original_category] = [
                {"effective_argument": x} for x in df["text"].values.tolist()
            ]

        return examples_per_category

    # GENERATION #
    # ---------- #
    def generate_for_prompts(
        self, ineffective_argument: str, category: str = None
    ):
        result_dict = {}
        # Preparing PROMPTS

        assert (self.n_shots > 0 and category is not None) or self.n_shots == 0

        local_examples = (
            [self.examples[category].pop() for _ in range(0, self.n_shots)]
            if self.n_shots > 0
            and self.examples is not None
            and len(self.examples[category]) > 0
            else []
        )

        for k, instructions in self.prompt_dict.items():
            if self.n_shots > 0:
                if len(local_examples) == 0:
                    print("ERROR!!! No examples left!!!")
                example_prompt = PromptTemplate(
                    input_variables=["effective_argument"],
                    template="This is an example of an effective argument: {effective_argument}",
                )

                prompt = FewShotPromptTemplate(
                    examples=local_examples,
                    example_prompt=example_prompt,
                    suffix=self.llm_model.get_template(instructions),
                    input_variables=["text"],
                )
            else:  # 0 shot
                prompt = self.llm_model.get_prompt_template(instructions)

            # remove
            if self._temp_flag:
                print("****** prompt: ")
                print(prompt.format(text=ineffective_argument))
                self._temp_flag = False

            llm_chain = LLMChain(llm=self.llm_model.llm, prompt=prompt)
            result_dict[k] = llm_chain.run(text=ineffective_argument)
            result_dict[f"len_{k}"] = len(result_dict[k])
            result_dict["len_orig"] = len(ineffective_argument)
            if self.n_shots > 0:
                result_dict["examples"] = local_examples

        return result_dict

    def generate_all(self, limit: int = -1):
        out_file = f"{self.out_file}{self.ideology}_{self.llm_model.name.lower()}_{self.n_shots}shot{self.out_ext}.jsonl"

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
            counter = 0
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
                    print(f"exception {e}")
                    print(
                        f"Failed to get a response for ID: {datapoint['idx']}"
                    )
                counter = counter + 1
                if counter >= limit and limit > -1:
                    break
