from iesta.llms.generate import Generator
import argparse


def run(ideology):

    generator = Generator(ideology=ideology,
                          model_name=Generator._MODEL_ALPACA_)
    generator.generate_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ideology", type=str)
    args = parser.parse_args()

    run(args.ideology)

# nohup python3 scripts/generate_w_alpaca.py -i liberal  > logs/alpaca_0shot.log 2>&1 &
# nohup python3 scripts/generate_w_alpaca.py -i conservative  > logs/cons_alpaca_0shot.log 2>&1 &
