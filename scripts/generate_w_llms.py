from iesta.llms.generate import Generator
import argparse
from dotenv import load_dotenv, find_dotenv
from iesta.llms.models import LlamaV2, ChatGpt

def run(ideology, models, examples_k, limit):
    models = ["llamav2", "chatgpt"] if models == "all" else [models]
    for model in models:
        iestal_model = LlamaV2() if model == "llamav2" else ChatGpt()

        for k in range(0, examples_k+1):
            print(f" #### For {k}shot prompts for model {model}, for {ideology} ###")

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
    parser.add_argument("-k", "--examples_k", type=int, default=0) #  shots
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

# nohup python3 scripts/generate.py -i all -k 1 -m chatgpt -l 2  > logs/all_fewshots.log 2>&1 &
