import dataclasses
import pandas as pd


@dataclasses.dataclass
class Evaluator:
    model_type: str  # either chatgpt or llamav2
    ideology: str
    shot_num: int

    ## exceptional - LIWC features_path
    feature_liwc_path: str = None
    root_path = "../data/llms_out/new"

    def __post_init__(self):
        self.filename: str = f"{self.root_path}/{self.ideoloy}_{self.model_type}_{self.shot_num}shot.jsonl"
        self.data = self.clean_data(self.filename)

    def clean_data(self):
        # ChatGpt
        data = pd.read_json(self.filename)
        if self.model_type == "chatgpt":
            pass
        # LLama
        elif self.model_type == "llamav2":
            pass
        return data

    def classify_effectiveness(self):
        #  Load model from hf and classify
        pass

    def extract_style_features(self):
        # 1. extract stykle features
        # 2. save significance
        pass

    def score_style():
        pass

    def score_morality():
        # like malitiousness
        # Expected: LLM significantly decrease it
        pass

    def score_stance():
        # percent of correct ones
        pass
