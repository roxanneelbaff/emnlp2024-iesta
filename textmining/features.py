

import gensim
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def _apply_basic_features(row, extracted_df):
    if row.name  in extracted_df.index :
        r = extracted_df.loc[row.name]
        for k, v in r.items():
            row[k] = v
    return row


# returns the N-grams frequency for one aspect (e.g. pos) in a dataframe:
# parameters:
##  df: dataframe that contains the content
## 'feature': str, the column name in the df that holds the data on which the 1-3 grams will be extracted
## 'training_ids': array, contains the IDs of the training data where the Vectorizer will be fit
## 'min_df': occurrence of n-gram in at least n documents where n can be an int or between ]0, 1[
## 'max_df': occurrence of n-gram in at most n documents where n can be an int or between ]0, 1[
## 'ngram_range': a tuple that contains the range of n-grams. e.g., (1,3) extract freq for 1 to 3 grams
## 'count_type': counter or 'tf-idf'
## 'idx': the column name of the dataframe that contains the id. Default: 'id'
## 'cols_prefix': return the features with column name {cols_prefix}_
def extract_n_grams_features(df, df_train, feature,  
                             min_df=30, max_df=0.4, ngram_range=(1,3),
                             count_type='counter', idx= 'id',
                            cols_prefix=''): #pos stem token
    df_original =  df.copy()
        
    ## fit on training
    #df_train= (df[df[idx].isin(training_ids)]).copy()
    
    df_ = df.copy()
    df_ = df_.reset_index()
    extracted_df = pd.DataFrame()
    
    # Initializing vectorizer
    vectorizer = None
    if count_type == 'tf-idf':
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range )
    elif count_type == 'counter':
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range )
        
    # Fitting in training data
    vectorizer.fit(df_train[feature])
    features = vectorizer.transform(df_[feature])

    extracted_df =pd.DataFrame(
        features.todense(),
        columns=vectorizer.get_feature_names()
    )
    extracted_df = extracted_df.add_prefix(cols_prefix)

    # Merging results with original df
    aid_df = df_[[idx]]

    extracted_df = extracted_df.merge(aid_df, left_index =True, right_index=True, suffixes=(False, False), how='inner')
    extracted_df.set_index(idx, inplace=True)

    result_df = df_original.apply(_apply_basic_features, axis=1, args=(extracted_df,))
    return result_df, extracted_df 