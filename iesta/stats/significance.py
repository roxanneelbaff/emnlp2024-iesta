import nlpaf.inferential_stats
import nlpaf.data_processor
import pandas as pd
import iesta.properties as prop
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from nlpaf.ml import preprocess
import os

ROOT_PATH = os.path.join(prop.ROOT_PATH, "significant_test")

ideologies = [
    prop.CONSERVATIVE_IDEOLOGY.lower(),
    prop.LIBERAL_IDEOLOGY.lower(),
]

# HELPERS


def _get_full_filename(
    ideology: str, independent_var: str, normalize: str = None
):
    normalized_str = f"_{normalize.lower()}" if normalize is not None else ""
    filename = "{}{}_{}".format(ideology, normalized_str, independent_var)
    file_path: str = os.path.join(ROOT_PATH, filename)

    return file_path, filename


def calc_sign_effects(
    original_df: pd.DataFrame,
    ideology: str,
    independent_var: str = "effect",
    normalize: str = None,
    recalculate: bool = False,
) -> pd.DataFrame:
    filepath, filename = _get_full_filename(
        ideology, independent_var, normalize
    )
    result_df, detailed_df = (
        nlpaf.inferential_stats._siginificance_calculated(filepath)
        if not recalculate
        else (None, None)
    )
    if result_df is not None and detailed_df is not None:
        return result_df, detailed_df

    df: pd.DataFrame = original_df.copy()

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_cols = df.select_dtypes(include=numerics).columns.to_list()
    num_cols.remove("round")
    cols = num_cols + [independent_var]
    print(f"normalizing with {normalize}")
    if normalize is not None:
        print("normalizing...")
        if normalize == "MinMaxScaler":
            # clip outliers
            print(f"Before clipping {len(df)}")
            df, _ = preprocess.clip_outliers(df)
            print(f"After clipping {len(df)}")
        scaler = (
            RobustScaler() if normalize == "RobustScaler" else MinMaxScaler()
        )
        df[num_cols] = scaler.fit_transform(df[num_cols])

    assert len(df[independent_var].unique()) > 1
    assert independent_var == "effect" or independent_var == "binary_effect"

    processed_df = df[cols].copy()

    #    processed_df = nlpaf.data_processor.undersample(
    #       df[cols],
    #        replacement=False,
    #        random_state=42,
    #        sampling_strategy="not minority",
    #    )
    #    print(f"processed df length: {len(processed_df)}")

    assert len(processed_df) > 0
    print(f"filepath {filepath}")
    return nlpaf.inferential_stats.significance(
        processed_df, filename=filepath, independent_var=independent_var
    )


def run_all_significance_test(feature_dfs):
    independent_var = "effect"

    significance = {}
    for ideology in ideologies:
        for normalize in [None, "RobustScaler", "MinMaxScaler"]:
            print(f"Running {ideology} - normalize:{normalize}")

            features_df = feature_dfs[ideology]

            significance[
                _get_full_filename(ideology, independent_var, normalize)[1]
            ] = calc_sign_effects(
                features_df, ideology, independent_var, normalize=normalize
            )

    return significance
