from sklearn.preprocessing import QuantileTransformer
import pandas as pd
from iesta.data.iesta_data import load_training_features_df


def get_top_features(ideology: str, root: str = ""): 
    #print(f"/content/drive/My Drive/Colab Notebooks/eacl2024/data/significant_test/summary_training_effect.csv")
    sign_features_df = pd.read_csv(
        f"{root}data/significant_test/summary_training_effect.csv"
    )
    sign_features_df = sign_features_df[
        (sign_features_df["label"] == ideology)
        | (sign_features_df["label"] == "same")
        | (sign_features_df["label"] == "opposite")
    ]
    col = f"absolute_effect_{ideology}"
    sign_features_df = sign_features_df.sort_values(by=[col], ascending=False)
    return sign_features_df[
        ["feature", f"effective ineffective_{ideology}", "label"]
    ]


def _add_score_features(row, ideology, top_features):
    feature_col_name = f"effective ineffective_{ideology}"
    features_score: float = 0.0
    for _, feature_row in top_features.iterrows():
        feature_name = feature_row["feature"]
        features_score = features_score + (
            row[feature_name] * feature_row[feature_col_name] * 1.0
        )
    row["features_score"] = features_score
    return row


def score_training_features(ideology: str) -> pd.DataFrame:
    top_features: pd.DataFrame = get_top_features(ideology)
    _, data = load_training_features_df(ideology.capitalize())

    scaler = QuantileTransformer()
    columns_to_normalize = top_features["feature"].values.tolist()
    data[columns_to_normalize] = data[columns_to_normalize].fillna(0.0)
    data[columns_to_normalize] = scaler.fit_transform(
        data[columns_to_normalize]
    )

    data = data.apply(
        _add_score_features,
        axis=1,
        args=(
            ideology,
            top_features,
        ),
    )

    data = data.sort_values(by=["features_score"], ascending=False)
    return data


def generate_training_w_scores(ideology: str = "liberal", save: bool = True):
    result = {}
    try:
        result["effective"] = pd.read_parquet(
            f"data/training/strategy_feature_based/{ideology}_effective.parquet"
        )
        result["ineffective"] = pd.read_parquet(
            f"data/training/strategy_feature_based/{ideology}_ineffective.parquet"
        )
    except Exception:
        df_score = score_training_features(ideology)

        for effect in ["effective", "ineffective"]:
            df = df_score[df_score["effect"] == effect]
            df = df[["effect", "cleaned_text", "features_score"]]
            ascending = effect == "ineffective"
            print(f"Saving for {effect} with ascending {ascending}")
            df = df.sort_values(by=["features_score"], ascending=ascending)
            if save:
                df.to_parquet(
                    f"data/training/strategy_feature_based/{ideology}_{effect}.parquet"
                )
            result[effect] = df

    return result
