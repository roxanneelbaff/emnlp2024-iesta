def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32 ]),
        }


all_configs = {}
_LIBERAL_SC_ = "style_evaluator_liberal"

_CONSERVATIVE_EVALUATOR = "style_evaluator_conservative_3052023"
_LIBERAL_EVALUATOR_ = "style_evaluator_liberal_3052023"
_LIBERAL_EVALUATOR1_ = "style_evaluator_liberal1"

_LIBERAL_EVALUATOR_HS_ = "style_evaluator_liberal_hs_3052023"

all_configs[_LIBERAL_EVALUATOR_] = {
    "ideology": "liberal",
    "undersample":True,
    "pretrained_model_name":"distilbert-base-uncased", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir":_LIBERAL_EVALUATOR_,
    "learning_rate": 4e-6,
    "per_device_train_batch_size":8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "is_for_style_classifier": True,
    "push_to_hub": True,
    "search_hp":False,
    "optuna_hp_func": None, #optuna_hp_space
}

all_configs[_LIBERAL_EVALUATOR_HS_] = {
    "ideology": "liberal",
    "undersample":True,
    "pretrained_model_name":"distilbert-base-uncased", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir":_LIBERAL_EVALUATOR_HS_,
    "learning_rate": 4e-6,
    "per_device_train_batch_size":8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "is_for_style_classifier": True,
    "push_to_hub": True,
    "search_hp":True,
    "optuna_hp_func": optuna_hp_space
}



all_configs[_CONSERVATIVE_EVALUATOR] = {
    "ideology": "conservative",
    "undersample":True,
    "pretrained_model_name":"distilbert-base-uncased", #"microsoft/deberta-base",
    "uncase": True,
    "output_dir":_CONSERVATIVE_EVALUATOR,
    "learning_rate": 4e-6,
    "per_device_train_batch_size":8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "is_for_style_classifier": True,
    "push_to_hub": False,
    "search_hp":False,
    "optuna_hp_func": None, #optuna_hp_space
}

all_configs[_LIBERAL_EVALUATOR1_] = {
    #"dataset_name": "notaphoenix/shakespeare_dataset",
    "ideology": "liberal",
    "undersample": True,
    "pretrained_model_name": "allenai/longformer-base-4096", #"microsoft/deberta-base",
    "uncase": False,
    "output_dir": f"{_LIBERAL_EVALUATOR1_}",
    "learning_rate": 2e-6,
    "per_device_train_batch_size" :8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 12,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "push_to_hub": False,
    "search_hp": False,
    "hub_private_repo": False,
    "optuna_hp_func": None,
    "is_for_style_classifier": True
}