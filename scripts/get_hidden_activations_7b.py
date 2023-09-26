import transformers
import torch
import pandas as pd
import glob
import pickle
from tqdm import tqdm
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

model_size = "7b_chat"
PATH_TO_ACTIVATION_STORAGE = "/hpc_data/kone_ka/EmEx/iesta_data/new_prompts"
root_dir = "/localdata2/kone_ka/eacl2024-iesta/"
MODEL_PATH = f"/hpc_data/kone_ka/EmEx/hf_models/llama2_{model_size}_hf/"

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
device = torch.device('cuda:0')
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     )

def get_training_prompt_data(ideology, without_system_prompt:bool = True, root_dir="../"):
    system_st = "without_system_prompt" if without_system_prompt else "with_system_prompt"
    effective = pd.read_parquet(f"{root_dir}data/training/{ideology}_{system_st}_effective.parquet")
    ineffective = pd.read_parquet(f"{root_dir}data/training/{ideology}_{system_st}_ineffective.parquet")
    
    return effective, ineffective

# SETTING 1: System Prompt is NOT included (you are a helpful...)
liberal_effective, liberal_ineffective = get_training_prompt_data("liberal", without_system_prompt=True, root_dir=root_dir)
conservative_effective, conservative_ineffective = get_training_prompt_data("conservative", without_system_prompt=True, root_dir=root_dir)

# SETTING 2: System Prompt is included (you are a helpful...)
liberal_w_system_effective, liberal_w_system_ineffective = get_training_prompt_data("liberal", without_system_prompt=False, root_dir=root_dir)
conservative_w_system_effective, conservative_w_system_ineffective = get_training_prompt_data("conservative", without_system_prompt=False, root_dir=root_dir)

data_names = ['liberal_effective', 'liberal_ineffective', 'conservative_effective', 'conservative_ineffective',
'liberal_w_system_effective', 'liberal_w_system_ineffective', 'conservative_w_system_effective', 'conservative_w_system_ineffective']

n = 0

for data in [liberal_effective, liberal_ineffective, conservative_effective, conservative_ineffective,
             liberal_w_system_effective, liberal_w_system_ineffective, conservative_w_system_effective, conservative_w_system_ineffective]:
    
    actis = []
    i = 0
    j = 0

    for index, row in tqdm(data.iterrows()):
        print(f"Generating activations for {data_names[n]}")
        n = n + 1 

        sentence = row['prompt']
        # print(sentence)
        input_tokens = tokenizer(sentence, return_tensors="pt").to(device)

        gen_text = model.forward(input_tokens.input_ids, output_hidden_states=True, return_dict=True)
        
        hidden_states = []
        #iterating over all layers and storing activations of the last token
        for layer in gen_text['hidden_states']:
            hidden_states.append(layer[0][-1].detach().cpu().numpy())

        actis.append([index, sentence, hidden_states, row['effect']])

        i += 1
        # save activations in batches
        if i == 10000:
            i = 0
            with open(f'{PATH_TO_ACTIVATION_STORAGE}/{data_names[n]}_activations_{j}.pkl', 'wb') as f:
                pickle.dump(actis, f)
            del actis
            del hidden_states
            actis = []
            j += 1

    # in case the number of samples is not dividable by 10000, we safe the rest
    with open(f'{PATH_TO_ACTIVATION_STORAGE}/{data_names[n]}_activations_{j}.pkl', 'wb') as f:
        pickle.dump(actis, f)
    del actis
    del hidden_states
