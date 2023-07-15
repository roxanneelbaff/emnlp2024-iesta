from iesta.llms.generate import Generator
import argparse


def run(ideology, models, fewshots, examples_k, with_similarity):
    if fewshots:
        for k in range(3):
            generator = None
            print(f" #### FOR K {k} ####")
            k = k+1
            if models == "all":
                generator = Generator(ideology=ideology,
                                      model_name=Generator._MODEL_ALPACA_,
                                      trainingdata_profiling=True,
                                      use_fewshots=fewshots,
                                      fewshots_num_examples=examples_k,
                                      fewshots_w_semantic_similarity=with_similarity
                                      )
                generator.generate_all()
                generator = None
                generator = Generator(ideology=ideology,
                                      model_name=Generator._MODEL_CHATGPT_,
                                      trainingdata_profiling=True,
                                      use_fewshots= fewshots,
                                      fewshots_num_examples=examples_k,
                                      fewshots_w_semantic_similarity=with_similarity
                                      )
                generator.generate_all()
            else:
                generator = Generator(ideology=ideology,
                                      model_name=models,
                                      trainingdata_profiling=True,
                                      use_fewshots= fewshots,
                                      fewshots_num_examples=examples_k,
                                      fewshots_w_semantic_similarity=with_similarity
                                      )
                generator.generate_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ideology", type=str, default="all")
    parser.add_argument("-f", "--fewshots", action='store_true')
    parser.add_argument("-k", "--examples_k", type=int, default=1)
    parser.add_argument("-s", "--with_similarity", action='store_true')
    parser.add_argument("-m", "--models", type=str, default="all")
    args = parser.parse_args()
    print(args)
    if args.ideology == "all":
        print("FOR LIBERALS")
        run("liberal", args.fewshots, args.examples_k, args.with_similarity)
        print("FOR CONSERVATIVES")
        run("conservative", args.fewshots, args.examples_k, args.with_similarity)
    else:
        run(args.ideology, args.models, args.fewshots, args.examples_k, args.with_similarity)

# nohup python3 scripts/generate_w_alpaca.py -i liberal  > logs/alpaca_0shot.log 2>&1 &
# nohup python3 scripts/generate_w_alpaca.py -i conservative  > logs/cons_alpaca_0shot.log 2>&1 &


# # nohup python3 scripts/generate_w_alpaca.py -i all -f -k 3 -m alpaca  > logs/alpaca_fewshots.log 2>&1 &
