def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-8, 5e-4, log=True
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16, 32]
        ),
    }


all_configs = {}
_LIBERAL_SC_ = "style_evaluator_liberal"

_CONSERVATIVE_EVALUATOR = "style_evaluator_conservative_3052023"
_LIBERAL_EVALUATOR_ = "style_evaluator_liberal_3052023"

_LIBERRAL_BB_ = "style_evaluator_liberal_bigbird"
_LIBERAL_EVALUATOR_HS_ = "style_evaluator_liberal_hs_3052023"


_liberal_deberta_ = "style_evaluator_liberal_debertav3"


_LIBERAL_EVALUATOR_LF_ = "liberal_longformer"
_CONSERVATIVE_EVALUATOR_LF_ = "conservative_longformer"


_LIBERAL_EVALUATOR_DEBERTA_ = "liberal_deb"
_CONSERVATIVE_EVALUATOR_DEBERTA_ = "conservative_deb"


all_configs[_LIBERAL_EVALUATOR_] = {
    "ideology": "liberal",
    "undersample": True,
    "pretrained_model_name": "distilbert-base-uncased",  # "microsoft/deberta-base",
    "uncase": True,
    "output_dir": _LIBERAL_EVALUATOR_,
    "learning_rate": 4e-6,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "is_for_style_classifier": True,
    "push_to_hub": True,
    "search_hp": False,
    "optuna_hp_func": None,  # optuna_hp_space
}

all_configs[_LIBERAL_EVALUATOR_HS_] = {
    "ideology": "liberal",
    "undersample": True,
    "pretrained_model_name": "distilbert-base-uncased",  # "microsoft/deberta-base",
    "uncase": True,
    "output_dir": _LIBERAL_EVALUATOR_HS_,
    "learning_rate": 4e-6,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "is_for_style_classifier": True,
    "push_to_hub": True,
    "search_hp": True,
    "optuna_hp_func": optuna_hp_space,
}


all_configs[_CONSERVATIVE_EVALUATOR] = {
    "ideology": "conservative",
    "undersample": True,
    "pretrained_model_name": "distilbert-base-uncased",  # "microsoft/deberta-base",
    "uncase": True,
    "output_dir": _CONSERVATIVE_EVALUATOR,
    "learning_rate": 4e-6,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "is_for_style_classifier": True,
    "push_to_hub": False,
    "search_hp": False,
    "optuna_hp_func": None,  # optuna_hp_space
}

all_configs[_LIBERAL_EVALUATOR_LF_] = {
    "ideology": "liberal",
    "undersample": True,
    "pretrained_model_name": "allenai/longformer-base-4096",  # reformer-crime-and-punishment "google/google/bigbird-roberta-base"allenai/longformer-base-4096", #"microsoft/deberta-base",
    "uncase": False,
    "output_dir": f"{_LIBERAL_EVALUATOR_LF_}",
    "learning_rate": 5e-6,  # /4.0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 6,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": True,
    "search_hp": False,
    "hub_private_repo": True,
    "optuna_hp_func": None,
    "is_for_style_classifier": True,
    "optimizer": "adamw_hf",
    "tokenizer_max_length": 1024,
    "tokenizer_padding": True,
    "tokenizer_special_tokens": None,  # {'pad_token': '[EOS]'},
    "mixed_precision": "fp16",
    "seed": 42,
}

all_configs[_CONSERVATIVE_EVALUATOR_LF_] = {
    "ideology": "liberal",
    "undersample": True,
    "pretrained_model_name": "allenai/longformer-base-4096",  # reformer-crime-and-punishment "google/google/bigbird-roberta-base"allenai/longformer-base-4096", #"microsoft/deberta-base",
    "uncase": False,
    "output_dir": f"{_CONSERVATIVE_EVALUATOR_LF_}",
    "learning_rate": 5e-6,  # /4.0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 6,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": True,
    "search_hp": False,
    "hub_private_repo": True,
    "optuna_hp_func": None,
    "is_for_style_classifier": True,
    "optimizer": "adamw_hf",
    "tokenizer_max_length": 1024,
    "tokenizer_padding": True,
    "tokenizer_special_tokens": None,  # {'pad_token': '[EOS]'},
    "mixed_precision": "fp16",
    "seed": 42,
}

all_configs[_LIBERAL_EVALUATOR_DEBERTA_] = {
    "ideology": "liberal",
    "undersample": True,
    "pretrained_model_name": "microsoft/deberta-v3-base",  # reformer-crime-and-punishment "google/google/bigbird-roberta-base"allenai/longformer-base-4096", #"microsoft/deberta-base",
    "uncase": False,
    "output_dir": f"{_LIBERAL_EVALUATOR_DEBERTA_}",
    "learning_rate": 5e-6,  # /4.0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 100,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": True,
    "search_hp": False,
    "hub_private_repo": True,
    "optuna_hp_func": None,
    "is_for_style_classifier": True,
    "optimizer": "adamw_hf",
    "tokenizer_max_length": 1024,
    "tokenizer_padding": True,
    "tokenizer_special_tokens": None,  # {'pad_token': '[EOS]'},
    "mixed_precision": "fp16",
    "seed": 42,
}
# google/bigbird-roberta-base


## optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_hf"`):
#          The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or
#          adafactor.
