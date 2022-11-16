import pandas as pd

import iesta.properties as prop
import iesta.processor as proc
import iesta.utils as utils

import textmining.lexicons as lexicons

from tqdm import tqdm

## HELPERS

def _add_id(row):
    row['numeric_id'] = int(row['Filename'].split('_')[1].split('.')[0])
    return row

## END OF HELPERS
def extract_liwc(ideology):
    file_path = prop.FEATURE_LIWC_PATH.format(ideology.lower())
    tqdm.pandas()
    if utils.file_exists(file_path):
        return pd.read_parquet(file_path)

    liwc_df = pd.read_csv(prop.RAW_LIWC_PATH.format(ideology.lower()))

    liwc_df = liwc_df.apply(_add_id, axis=1)
    liwc_df.set_index('numeric_id', inplace=True)

    liwc_df = liwc_df.add_prefix('liwc_')
    liwc_df.drop(['liwc_Filename'], axis=1, inplace=True)

    liwc_df.to_parquet(file_path, index=False)
    return liwc_df

def extract_nrc_emotion(ideology):
    file_path = prop.FEATURE_NRC_PATH.format(ideology.lower())

    if utils.file_exists(file_path):
        return pd.read_parquet(file_path)

    print('file not saved, getting original data...')
    processor = proc.Process()
    arguments_df, _ = processor.get_ideology_based_voter_participant_df(ideology)

    nrc_df = arguments_df[['argument']].copy()

    print('calculating lexicon cound for nrc...')
    nrc_df = lexicons.count_nrc_emotions_and_sentiments(nrc_df, text_column='argument')
    nrc_df.to_parquet(file_path, index=False)
    return nrc_df

def extract_mpqa_arg(ideology):
    file_path = prop.FEATURE_MPQA_ARG_PATH.format(ideology.lower())

    if utils.file_exists(file_path):
        return pd.read_parquet(file_path)

    print('file not saved, getting original data...')
    processor = proc.Process()
    arguments_df, _ = processor.get_ideology_based_voter_participant_df(ideology)

    mpqa_arg_df = arguments_df[['argument']].copy()

    print('calculating lexicon cound for mpqa arg...')
    mpqa_arg_df = lexicons.count_mpqa_arg(mpqa_arg_df, text_column='argument',prefix='mpqa_arg_')
    mpqa_arg_df.to_parquet(file_path, index=False)
    return mpqa_arg_df


def extract_empath(ideology):
    file_path = prop.FEATURE_EMPATH_PATH.format(ideology.lower())

    if utils.file_exists(file_path):
        return pd.read_parquet(file_path)

    print('file not saved, getting original data...')
    processor = proc.Process()
    arguments_df, _ = processor.get_ideology_based_voter_participant_df(ideology)

    empath_df = arguments_df[['argument']].copy()

    print('calculating lexicon cound for empath...')
    empath_df = lexicons.count_empath(empath_df, text_column='argument',prefix='empath_')
    empath_df.to_parquet(file_path, index=False)
    return empath_df


#count_empath_for_ideologies
def extract_empath_ideology(ideology):
    file_path = prop.FEATURE_EMPATH_IDEOLOGY_PATH.format(ideology.lower())

    if utils.file_exists(file_path):
        return pd.read_parquet(file_path)

    print('file not saved, getting original data...')
    processor = proc.Process()
    arguments_df, _ = processor.get_ideology_based_voter_participant_df(ideology)

    empath_df = arguments_df[['argument']].copy()
    print('data has: ', len(empath_df))
    print('calculating lexicon cound for empath...')
    empath_df = lexicons.count_empath_for_ideologies(empath_df, text_column='argument',prefix='empath_ideology_')
    empath_df.to_parquet(file_path, index=False)
    return empath_df