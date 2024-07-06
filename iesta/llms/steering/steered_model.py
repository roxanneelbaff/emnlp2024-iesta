import iesta.llms.steering.config as config
import pickle
import torch
import numpy as np
from torch import nn
import transformers
from iesta.llms.steering.steering_layer import SteeringLayer
import dataclasses
from types import FunctionType
from dataclasses import field

@dataclasses.dataclass
class SteeredModel:
    model_name_path: str = config.MODEL_NAME_PATH
    activation_file_paths: list[str] = field(default_factory=config.ACTIVATION_FILE_PATHS) 
    concat_func: FunctionType = np.mean

    def __post_init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name_path
        )
        self.device = torch.device("cuda:0")

        print("Loading activation layers...")
        self.activation_layer_per_syle_dict = self.load_activations()

        print("Concatinating each style hidden layers...")
        self.concatenated_style_dict = self.concatenate_style(self.concat_func)

        print("Calculating steering vectors...")
        self.steering_vectors_per_style_dict = self.get_steering_vectors()

        print(
            "Each of these 'styles' have a steering vector, here are the keys:"
            f" {self.steering_vectors_per_style_dict.keys()}"
        )

    def load_activations(self) -> dict:
        activation_layer_per_syle_dict = {
            k: [] for k in config.get_style_experiment_keys()
        }
        for key, _ in activation_layer_per_syle_dict.items():
            style_files = [
                f
                for f in self.activation_file_paths
                if f.split("/")[-1].startswith(key)
            ]
            print(f"style files: {style_files}")
            for style_file in style_files:
                with open(style_file, "rb") as f:
                    activation_pickle = pickle.load(f)
                    for acti in activation_pickle:
                        activation_layer_per_syle_dict[key].append(acti[2][18:21])  # 0 is the ID and 1 is the sample as a string 2 holds all the layers
        return activation_layer_per_syle_dict

    def concatenate_style(self, func=np.mean) -> dict:
        concatenated_style_dict = {}
        for key, value in self.activation_layer_per_syle_dict.items():
            concatenated_style_dict[key] = []
            for i in range(len(config.INSERTION_LAYERS)):
                concatenated_style_dict[key].append(func(value[i], 0))
        return concatenated_style_dict

    @staticmethod
    def _get_np_array(_arr):
        return np.array([item for sublist in _arr for item in sublist])

    def get_steering_vectors(self):
        np_arr_per_style_dict = {
            k: [] for k in config.get_style_experiment_keys()
        }
        np_arr_per_style_dict = {
            k: SteeredModel._get_np_array(self.concatenated_style_dict[k])
            for k, _ in np_arr_per_style_dict.items()
        }

        steering_vectors = {
            k: np.split(
                np_arr_per_style_dict[k]
                - np_arr_per_style_dict[config.get_opposite_style(k)],
                len(config.INSERTION_LAYERS),
            )
            for k in config.get_style_experiment_keys()
        }
        return steering_vectors

    def get_steered_model(self, style_key: str, lmbda: float):
        steering_vectors = self.steering_vectors_per_style_dict[style_key]

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name_path, device_map="auto", torch_dtype=torch.float16
        )

        for n, _ in enumerate(config.INSERTION_LAYERS):
            model.model.layers[config.INSERTION_LAYERS[n]].mlp = SteeringLayer(
                model.model.layers[config.INSERTION_LAYERS[n]].mlp
            )
            model.model.layers[
                config.INSERTION_LAYERS[n]
            ].mlp.steering_vector = nn.Parameter(
                torch.from_numpy(steering_vectors[n]).to(self.device)
            )
            model.model.layers[
                config.INSERTION_LAYERS[n]
            ].mlp.b = lmbda  # "meta-llama/Llama-2-7b-chat-hf"
        return model

    def test_generate(self, model, inputs: list):
        for _, sentence in enumerate(inputs):
            input_tokens = self.tokenizer(sentence, return_tensors="pt").to(
                self.device
            )

            gen_tokens = model.generate(
                input_tokens.input_ids, max_new_tokens=1024
            )

            output = (
                self.tokenizer.batch_decode(gen_tokens)[0]
                .replace(sentence, "")
                .replace("\n", " ")
                .replace(";", "-")
            )
            print(output)
            return output
