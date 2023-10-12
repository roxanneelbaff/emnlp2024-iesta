import pandas as pd

if __name__ == "__main__":
    liberal_parquet = "data/splitted/liberal_training.parquet"
    conservative_parquet = "data/splitted/conservative_training.parquet"

    pd.read_parquet(liberal_parquet).to_csv(
        liberal_parquet.replace("parquet", "csv")
    )
    pd.read_parquet(conservative_parquet).to_csv(
        conservative_parquet.replace("parquet", "csv")
    )
