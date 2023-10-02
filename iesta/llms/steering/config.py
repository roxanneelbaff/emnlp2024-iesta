import itertools
import glob
import torch


## IESTA SPECIFIC
PATH_TO_ACTIVATION_STORAGE: str = "data/training/hidden_layers"
INSERTION_LAYERS = [18, 19, 20]
ACTIVATION_FILE_PATHS = glob.glob(f"{PATH_TO_ACTIVATION_STORAGE}/*")
MODEL_NAME_PATH = "meta-llama/Llama-2-7b-chat-hf"


def get_style_experiment_keys() -> list:
    ideologies = ["liberal", "conservative"]
    with_system_prompt = ["", "w_system"]
    effectiveness = ["effective", "ineffective"]

    combinations = list(
        itertools.product(ideologies, with_system_prompt, effectiveness)
    )
    keys = ["_".join(k).replace("__", "_") for k in combinations]
    return keys


def get_opposite_style(key) -> str:
    base_name = "_".join(key.split("_")[:-1])
    style = key.split("_")[-1]
    opposite = "ineffective" if style == "effective" else "effective"
    return f"{base_name}_{opposite}"


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
