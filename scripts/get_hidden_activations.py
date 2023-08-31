import transformers
import torch
import pandas as pd
import glob
import pickle
from tqdm import tqdm

# model_size = "7b_chat"
model_name_path = "meta-llama/Llama-2-7b-chat-hf"
PATH_TO_ACTIVATION_STORAGE = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0003/emex/steering_vectors/eacl_iesta/7b-chat/"
PATH_TO_SAMPLES = "/dss/dsshome1/0A/di93qex/dev/eacl2024-iesta/data/training/strategy_feature_based/"
# MODEL_PATH = f"/dss/dsstbyfs02/pn49ci/pn49ci-dss-0003/emex/llama2/llama/llama2_{model_size}_hf/"

data = glob.glob(PATH_TO_SAMPLES + '*')

device = torch.device('cuda:0')
model = transformers.AutoModelForCausalLM.from_pretrained(model_name_path, device_map='auto', torch_dtype=torch.float16)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_path)

for frame in data:
    df = pd.read_parquet(frame)
    file_name = frame.split('/')[-1].split('.')[0]
    if file_name != 'conservative_ineffective':
        continue
    print(f"Generating activations for {file_name}")
    actis = []

    i = 0
    j = 0
    for index, row in tqdm(df.iterrows()):

        # removing newlines from samples.
        sentence = row['cleaned_text']
        input_tokens = tokenizer(sentence, return_tensors="pt").to(device)

        # if len(input_tokens.input_ids[0]) > 2440:
        #     continue
        # print(len(input_tokens.input_ids[0]))
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
            with open(f'{PATH_TO_ACTIVATION_STORAGE}/{file_name}_activations_{j}.pkl', 'wb') as f:
                pickle.dump(actis, f)
            del actis
            del hidden_states
            actis = []
            j += 1
            exit()

    # in case the number of samples is not dividable by 10000, we safe the rest
    with open(f'{PATH_TO_ACTIVATION_STORAGE}/{file_name}_activations_{j}.pkl', 'wb') as f:
        pickle.dump(actis, f)
    del actis
    del hidden_states
