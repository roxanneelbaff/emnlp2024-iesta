from transformers import LlamaTokenizer, LlamaForCausalLM
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
import transformers
import torch
import pandas as pd
import glob
import pickle
from tqdm import tqdm

model_size = "13b_chat"
PATH_TO_ACTIVATION_STORAGE = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0003/emex/steering_vectors/eacl_iesta/"
PATH_TO_SAMPLES = "/dss/dsshome1/0A/di93qex/dev/eacl2024-iesta/data/training/strategy_feature_based/"
MODEL_PATH = f"/dss/dsstbyfs02/pn49ci/pn49ci-dss-0003/emex/llama2/llama/llama2_{model_size}_hf/"

data = glob.glob(PATH_TO_SAMPLES + '*')

device = torch.device('cuda:0')
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

# sentence = """##Instructions: 
# Annotate all place names with geocoordinates.
# ##Data:
# First they will go to Krakow in Poland, then by road to the border, and finally onto the train that will carry them across Ukraine to the reunion with Jenia 
# - the father and husband they have missed so much every day and night of exile.
# Oksana says she cannot believe they will see him soon. "It's like a dream." Then she asks herself a question, and answers all at once: 
# "Can I believe it? Yes!"
# The story of exile from Ukraine begins in the darkness of 24 February 2022, 
# when the first Russian artillery shells began to land in the Kharkiv suburb of Saltivka. 
# The couple had been watching the news about a troop build-up just over the border, but like so many Ukrainians, 
# Oksana and Jenia wanted to protect their children from the fear of war.
# ##Geocoordinates:
# """
# sentence.replace("\n", " ")
# input_tokens = tokenizer(sentence, return_tensors="pt").to(device)
# gen_text = model.generate(input_tokens.input_ids, max_new_tokens=500)
# print(tokenizer.batch_decode(gen_text))

# print(len(input_tokens.input_ids[0]))

# device_map = infer_auto_device_map(model, max_memory={0: "70GiB", 1: "70GiB"})
# model = load_checkpoint_and_dispatch(
#     model, checkpoint=MODEL_PATH, device_map=device_map
# )

for frame in data:
    df = pd.read_parquet(frame)
    file_name = frame.split('/')[-1].split('.')[0]
    print(f"Generating activations for {file_name}")
    actis = []

    i = 0
    j = 0
    for index, row in tqdm(df.iterrows()):
    # for index, row in df.iterrows():

        # removing newlines from samples.
        sentence = row['cleaned_text'].replace('\n', ' ')
        input_tokens = tokenizer(sentence, return_tensors="pt").to(device)

        if len(input_tokens.input_ids[0]) > 2440:
            continue
        # print(len(input_tokens.input_ids[0]))
        gen_text = model.forward(input_tokens.input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = []
        # print("Generating Done")
        #iterating over all layers and storing activations of the last token
        for layer in gen_text['hidden_states']:
            hidden_states.append(layer[0][-1].detach().cpu().numpy())

        # shakespeare and yelp store the labels in column 'sentiment', go emotion stores labels in 'labels' column.
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

    # in case the number of samples is not dividable by 10000, we safe the rest
    with open(f'{PATH_TO_ACTIVATION_STORAGE}/{file_name}_activations_{j}.pkl', 'wb') as f:
        pickle.dump(actis, f)
    del actis
    del hidden_states
