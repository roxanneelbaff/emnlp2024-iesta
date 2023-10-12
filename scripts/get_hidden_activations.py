import transformers
import torch
import pandas as pd
import glob
import pickle
from tqdm import tqdm
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
    init_empty_weights,
)

model_size = "7b_chat"
PATH_TO_ACTIVATION_STORAGE = "../data/hidden_activations/"
PATH_TO_SAMPLES = "../data/training/strategy_feature_based/"
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"  # f"/hpc_data/kone_ka/EmEx/hf_models/llama2_{model_size}_hf/"

data = glob.glob(PATH_TO_SAMPLES + "*")

# tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, model_max_length=2048)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda:0")
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
)
# pipe = transformers.pipeline("text-generation",
#                         model=model,
#                         tokenizer=tokenizer,
#                         torch_dtype=torch.float16,
#                         device_map="auto",
#                         max_new_tokens=2048,
#                         do_sample=True,
#                         return_dict=True,
#                         output_hidden_states=True,
#                         top_k=30,
#                         num_return_sequences=1,
#                         eos_token_id=tokenizer.eos_token_id,
#                         )

for frame in data:
    df = pd.read_parquet(frame)
    file_name = frame.split("/")[-1].split(".")[0]
    print(f"Generating activations for {file_name}")
    actis = []

    i = 0
    j = 0
    for index, row in tqdm(df.iterrows()):
        # removing newlines from samples.
        # sentence = row['cleaned_text'].replace('\n', ' ')
        sentence = row["cleaned_text"].replace("<URL>", "")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            input_tokens = tokenizer(sentence, return_tensors="pt").to(device)

            gen_text = model.forward(
                input_tokens.input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            # gen_text = pipe(sentence)

            hidden_states = []
            # iterating over all layers and storing activations of the last token
            for layer in gen_text["hidden_states"]:
                hidden_states.append(layer[0][-1].detach().cpu().numpy())

            actis.append([index, sentence, hidden_states, row["effect"]])

            i += 1
            # save activations in batches
            if i == 10000:
                i = 0
                with open(
                    f"{PATH_TO_ACTIVATION_STORAGE}/{file_name}_activations_{j}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(actis, f)
                del actis
                del hidden_states
                actis = []
                j += 1

    # in case the number of samples is not dividable by 10000, we safe the rest
    with open(
        f"{PATH_TO_ACTIVATION_STORAGE}/{file_name}_activations_{j}.pkl", "wb"
    ) as f:
        pickle.dump(actis, f)
    del actis
    del hidden_states
