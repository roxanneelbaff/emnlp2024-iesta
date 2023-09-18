from iesta.llms.generate import Generator
import argparse
from dotenv import load_dotenv, find_dotenv
from iesta.llms.models import LlamaV2, ChatGpt


def run(ideology, models, examples_k, limit):
    models = ["llamav2", "chatgpt"] if models == "all" else [models]
    for k in range(0, examples_k+1):
        for model in models:
            iestal_model = LlamaV2(model_name_path="meta-llama/Llama-2-7b-chat-hf") if model == "llamav2" else ChatGpt()
            #hf= "meta-llama/Llama-2-13b-chat-hf"
            #server_llamav2 = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0003/emex/llama2/llama/llama2_13b_chat_hf/"
            #iestal_model = LlamaV2(server_llamav2) if model == "llamav2" else ChatGpt()
            print(f" #### {k}shot prompts - {model} - {ideology} ###")

            generator = Generator(
                ideology=ideology,
                llm_model=iestal_model,
                flag_profile_training_data=False,
                flag_profile_test_data=False, n_shots=k,
                root_path="data/"
            )
            generator.generate_all(limit=limit)


if __name__ == "__main__":
    found = load_dotenv(find_dotenv())
    print(f"dotenv was found: {found}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ideology", type=str, default="all")# liberal or conservative
    parser.add_argument("-k", "--examples_k", type=int, default=0) #  shots, 1 means run 0 and 1 shot
    parser.add_argument("-m", "--models", type=str, default="all") ## llamav2, chatgpt, + support models with activations
    parser.add_argument("-l", "--limit", type=int, default=-1) # number of text samples for ineffective arguments
    args = parser.parse_args()
    print(args)
    ideologies = [args.ideology]
    if args.ideology == "all":
        ideologies = ["liberal", "conservative"]

    for ideology in ideologies:
        run(
            ideology,
            args.models,
            args.examples_k,
            args.limit
        )

#  nohup python3 scripts/generate_w_llms.py  -k 0 -m llamav2  > logs/llamav20shot.log 2>&1 &
