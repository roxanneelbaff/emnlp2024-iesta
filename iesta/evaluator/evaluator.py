

import dataclasses
import pandas as pd

@dataclasses.dataclass
class Evaluator():
    data_path: str
    model_type: str
    ideology: str

    ## exceptional - LIWC features_path
    feature_liwc_path: str = None


    def __post_init__(self):
        self.data = self.clean_data()

    def clean_data(self):
        # ChatGpt
        data = pd.DataFrame()
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