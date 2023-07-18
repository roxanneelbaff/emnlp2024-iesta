import nlpaf.inferential_stats
import nlpaf.data_processor
import pandas as pd
from typing import List
import iesta.utils
import iesta.properties as prop
import iesta.stats.significance

import os 

ROOT_PATH = os.path.join(prop.ROOT_PATH, "significant_test")

ideologies = [
    prop.CONSERVATIVE_IDEOLOGY.lower(),
    prop.LIBERAL_IDEOLOGY.lower(),
]


def _get_full_filename(
    ideology: str,
    features_code: str,
    independent_var: str,
    undersample: bool = False,
):
    undersample_str = "_undersampled" if undersample else ""

    file_path: str = ROOT_PATH + "{}{}_{}_{}".format(
        ideology, undersample_str, features_code,  independent_var
    )

    return file_path


def calc_sign_effects(
    original_df: pd.DataFrame,
    ideology: str,
    features_code: str,
    independent_var: str = "effect",
    undersample: bool = False,
    recalculate: bool = False,
) -> pd.DataFrame:
    filename = _get_full_filename(
        ideology, features_code, independent_var, undersample)
    result_df, detailed_df = nlpaf.inferential_stats._siginificance_calculated(
        filename
    )
    if result_df is not None and detailed_df is not None:
        return result_df, detailed_df

    df: pd.DataFrame = original_df.copy()

    assert len(df[independent_var].unique()) > 1
    assert independent_var == "effect" or independent_var == "binary_effect"

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_cols = df.select_dtypes(include=numerics).columns.to_list()
    num_cols.remove("round")
    cols = num_cols + [independent_var]

    processed_df = df[cols].copy()

    if undersample:
        processed_df = nlpaf.data_processor.undersample(
            df[cols],
            replacement=False,
            random_state=42,
            sampling_strategy="not minority",
        )
        print(f"processed df length: {len(processed_df)}")

    assert len(processed_df) > 0
    return nlpaf.inferential_stats.significance(
        processed_df, filename=filename, independent_var=independent_var
    )


def run_all_significance_test(feature_dfs):
    independent_vars = ["effect"]
    #excluded_effects = [[], ["okay"]]

    undersample = [True, False]
    significance = {}
    for ideology in ideologies:
        for iv in independent_vars:
            # for excluded_effect in excluded_effects:
            for us in undersample:
                print("Running {ideology} {iv} {excluded_effect}")

                undersample_str = "_undersampled" if us else ""

                features_df = feature_dfs[ideology]
           
                significance[
                    f"{ideology}_{iv}{undersample_str}_all_features"
                ] = iesta.stats.significance.calc_sign_effects(
                    features_df,
                    ideology,
                    "all_features",
                    iv,
                    undersample=us,
                )

                    # transformers_features_df  = feature_dfs[ideology]["transformers"]
                    # if len(excluded_effect) >0:
                    #    transformers_features_df  = transformers_features_df[~transformers_features_df[iv].isin(excluded_effect)]
                    # significance[f'{ideology}_{iv}{undersample_str}{excluded_str}_transformers'] =  iesta.stats.significance.calc_sign_effects(transformers_features_df, ideology, "transformers", iv, undersample= us, exclude_iv_vals = excluded_effect)
    return significance
