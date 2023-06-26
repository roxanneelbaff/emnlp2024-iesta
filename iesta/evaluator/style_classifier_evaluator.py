import evaluate
from evaluate.evaluation_suite import SubTask
from iesta.machine_learning.huggingface_loader import IESTAHuggingFace

class Suite(evaluate.EvaluationSuite):

    def __init__(self, name):
        super().__init__(name)
        self.preprocessor = lambda x: {"text": x["text"].lower()}
        self.suite = [
            SubTask(
                task_type="text-classification",
                data="notaphoenix/",
                split="test",
                args_for_task={
                    "metric": "f1",
                    "average": "macro"
                    "input_column": "text",
                    "label_column": "label",
                    "label_mapping": {IESTAHuggingFace._LABEL2ID_}
                }
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="rte",
                split="validation[:10]",
                args_for_task={
                    "metric": "accuracy",
                    "input_column": "sentence1",
                    "second_input_column": "sentence2",
                    "label_column": "label",
                    "label_mapping": {
                        "LABEL_0": 0,
                        "LABEL_1": 1
                    }
                }
            )
        ]