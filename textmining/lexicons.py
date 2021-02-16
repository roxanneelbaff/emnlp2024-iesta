import pandas as pd
import spacy
from spacy_arguing_lexicon import ArguingLexiconParser
from empath import Empath


# download the NRC emotion lexicon from http://sentiment.nrc.ca/lexicons-for-research/NRC-Emotion-Lexicon.zip
# unzip it and put it under "lexicon", in the same folder as this file, OR
# put it wherever you want and provide the full path in the parameter 'filepath' of 
# "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
NRC_FILEPATH = "../lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
def load_nrc_emotions(filepath=NRC_FILEPATH ):
    emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"],  sep='\t')
    emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
    emolex_words.head()
    emolex_words.set_index(['word'], inplace=True)
    return emolex_df, emolex_words


def count_lexicons_one(text, categories, lexicon_words_df):
    # initialize
    counts_dict = {}
    total_words = 0
    found = 0
    for category in categories:
        counts_dict[category] = 0
        
    for word in text.split():
        total_words += 1
        if word in lexicon_words_df.index:
            for category in categories:
                found += 1 
                counts_dict[category] += lexicon_words_df.loc[word][category]
            
    counts_dict['ratio'] = round(float(found) / float(total_words), 5) if total_words >0 else 0
    return counts_dict
    
def _count_nrc_emotions_and_sentiments(row, emotions,sentiments, nrc_df,
                                        text_column = 'text', 
                                       path=NRC_FILEPATH, prefix='nrc_'):
    text = row[text_column]
    
    emotions_count_dict =count_lexicons_one(text, emotions, nrc_df)
    sentiments_count_dict =count_lexicons_one(text, sentiments, nrc_df)
    
    for k, v in emotions_count_dict.items(): 
        row[prefix+k] = v
        
    for k, v in sentiments_count_dict.items(): 
        row[prefix+k] = v

    return row


# 'df': The dataframe that contains the data
# 'text_column': The column name that contains the text that should be analyzed
# 'path': The path to the nrc lexicon. Default: "lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
# 'prefix': the prefic to attach to the column names that contain the results. Default: nrc
def count_nrc_emotions_and_sentiments(df, text_column = 'text', path=NRC_FILEPATH, prefix='nrc_'):
    nrc_df, nrc_words_df = load_nrc_emotions(path)
    
    emotions = list(nrc_df.emotion.unique() )
    emotions.remove('positive')
    emotions.remove('negative')
    sentiments = ['negative', 'positive']
    
    result_df = df.copy()
    result_df = result_df.apply(_count_nrc_emotions_and_sentiments, axis=1, args=(emotions,sentiments, nrc_words_df,
                                                                      text_column, path, prefix))
    return result_df
    
# MPQA Subjectivity. Download opinion finder v2.0
# Run command
# 
#'''
#cd ~/lexicons/opinionfinderv2.0
#java -Xmx1g -classpath ./lib/weka.jar:./lib/stanford-postagger.jar:opinionfinder.jar opin.main.RunOpinionFinder {path}/corpus.doclist -d

#'''
# Opinion finder data

def apply_mpqa_sentences_subjectivity(row, path='annotations/opinionfinder'):
    # annotation path
    opinion_finder_output = '{}/{}.txt_auto_anns/'
    sentence_subjectivity_file_path = opinion_finder_output + 'sent_subj.txt'
    
    # extract annotations
    doc_id = row.name
    
    annotation_path = sentence_subjectivity_file_path.format(path, doc_id)
    annotations =pd.read_csv(annotation_path, header=None, sep='\t', names=['id', 'sent_polarity'])
    
    # count obj and subj
    counts_dic = annotations['sent_polarity'].value_counts()
    polarity_sum = annotations['sent_polarity'].value_counts().sum()
    for key in counts_dic.keys():
        row['mpqa_subjobg_'+key] = float(counts_dic[key]) 

    
    # total 
    #path = 'corpus/{}.txt'.format(article_id)
    #text = ''# get text
    ##with open(path, 'r') as f:
    #    text = f.read()
    #total_sentences = len(sent_tokenize(text))
    # save to dataframe
    #row['mpqa_polarity_ratio'] = round(float(polarity_sum) / float(total_sentences ), 5)
    
    return row


def count_mpqa_subj_obj(df, annotations_path = 'annotations/opinionfinder'):
    df = df.apply(apply_mpqa_sentences_subjectivity, args=(annotations_path,), axis=1)
    return df



def load_arg_lexicon():
    arg_lexicon_labels = ['wants', 'contrast', 'assessments','doubt',
                          'authority','emphasis','necessity','causation',
                          'generalization','structure','conditionals',
                          'inconsistency','possibility','priority','difficulty',
                          'inyourshoes','rhetoricalquestion']

    arg_lex_nlp = spacy.load("en")
    arg_lex_nlp.add_pipe(ArguingLexiconParser(lang=arg_lex_nlp.lang))
    return arg_lex_nlp, arg_lexicon_labels

def _count_mpqa_arg(row, arg_lex_nlp, arg_lexicon_labels, text_column, prefix='mpqa_arg_'):
    
    text = row[text_column]
    doc = arg_lex_nlp(text)
    arguments = list(doc._.arguments.get_argument_spans_and_matches())
    total = len(arguments)
    
    # init vals 
    for label in arg_lexicon_labels:
        row[prefix+label] = 0
        
    # count lexicon
    total_arg_words = 0
    for arg in arguments:
        arg_span = arg[0]
        lexicon_label = arg_span.label_
        row[prefix+lexicon_label] += 1
        total_arg_words += len(arg_span.text)
    
    row['mpqa_arg_lexicon_ratio'] = round(float(total) / float( len(text.split())), 5)
    return row

def count_mpqa_arg(df, text_column='text',prefix='mpqa_arg_'):
    arg_lex_nlp, arg_lexicon_labels = load_arg_lexicon()
    result = df.apply(_count_mpqa_arg, axis=1, args=(arg_lex_nlp, arg_lexicon_labels, text_column, prefix))
    return result


## EMPATH from Stanford
def _count_empath_categories(row, text_column, empath_lexicon, prefix):
    empath_dic = empath_lexicon.analyze(row[text_column], normalize=True)
    for k,v in empath_dic.items():
        row['{}_{}'.format(prefix, k)] = v

    return row
def count_empath(df, text_column='text',prefix='empath_'):
    lexicon = Empath()
    result = df.apply(_count_empath_categories, axis=1, args=(text_column, lexicon, prefix))
    return result

