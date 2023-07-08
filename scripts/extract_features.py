
import argparse


from iesta.machine_learning.feature_extraction import extract_features
import iesta.properties as prop  
from iesta.machine_learning.dataloader import IESTAData, METHODOLOGY, LABELS


def main():
    try:

        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--ideology", type=str)
        parser.add_argument("-b", "--batch", type=int, default=300)
        parser.add_argument("-n", "--n_processors", type=int, default=1)
        parser.add_argument("-t", "--is_transformer", action='store_true')
    
        args = parser.parse_args()

        n_processes = args.n_processors
        if args.is_transformer and args.n_processors > 1:
            print("Warning! use 1 for processes "
                  f"instead of {args.n_processors} when using transformers")
            n_processes = 1
        print(args)

        dataloader: IESTAData = IESTAData(ideology=args.ideology,
                                          keep_labels=LABELS.EFF_INEFF,
                                          methodology=METHODOLOGY.EACH)
        _, path_ = dataloader.get_training_data(reload=False)
        extract_features(ideology=args.ideology,
                         data_path=path_,
                         batch_size=args.batch,
                         spacy_n_processors=n_processes,
                         transformer_based_features=args.is_transformer
                         )
    finally:
        pass


if __name__ == "__main__":
    main()


# Extract features for Liberals:
##   non-transformers: nohup python3 scripts/extract_features.py -i Liberal -b 5000 -n 5 > logs/feature_extraction_liberal.log  2>&1 &
##   transformers: nohup python3 scripts/extract_features.py -i Liberal -b 100 -n 1 -t > logs/feature_extraction_liberal.log  2>&1 &

# Extract features for Conservatives:
##   non-transformers: nohup python3 scripts/extract_features.py -i Conservative -b 5000 -n 5 > logs/feature_extraction_conservative.log  2>&1 &
##   transformers: nohup python3 scripts/extract_features.py -i Conservative -b 100 -n 1 -t > logs/feature_extraction_conservative.log  2>&1 &