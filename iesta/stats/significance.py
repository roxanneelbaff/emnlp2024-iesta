import nlpaf.inferential_stats
import pandas as pd
from typing  import List
import iesta.utils

ROOT_PATH ="../data/significant_test/"


def _get_signficance_df(ideology, features_code, independent_var, exclude_iv_vals, recalculate):

    excluded_str=""
    if exclude_iv_vals is not None and len(exclude_iv_vals)>0:
        excluded_str = f"_excluded-{'-'.join(exclude_iv_vals)}"


    file_path:str = ROOT_PATH+'{}_{}{}_{}'.format(ideology, features_code, excluded_str, independent_var)
    if iesta.utils.file_exists(file_path) and not recalculate:
        return file_path, pd.read_csv(f'{file_path}.csv')
    return file_path, None


def calc_sign_effects(original_df:pd.DataFrame, 
                        ideology:str, 
                        features_code:str, 
                        independent_var:str = "effect", 
                        exclude_iv_vals : List = [],
                        recalculate:bool = False):

    file_path, result_df= _get_signficance_df(ideology, features_code, independent_var, exclude_iv_vals, recalculate)
    if result_df is not None:
        return result_df


    df :pd.DataFrame = original_df.copy()

    if exclude_iv_vals is not None and len(exclude_iv_vals)>0:
        print(f"The IV has\n {df[independent_var].value_counts()}. Excluding {exclude_iv_vals}...")
        df = df[~df[independent_var].isin(exclude_iv_vals)]
        print(f"After exclusion: \n {df[independent_var].value_counts()}. Excluding {exclude_iv_vals}...")

    
    
    assert len(df[independent_var].unique()) >1
    assert independent_var == "effect" or  independent_var == "binary_effect"


    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_cols = df.select_dtypes(include=numerics).columns.to_list()
    num_cols.remove("round")
    cols = num_cols + [independent_var]

    
    return nlpaf.inferential_stats.significance(df[cols], 
                        save=True, 
                        desc=file_path,
                        independent_var=independent_var)