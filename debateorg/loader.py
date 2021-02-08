
import json
import pandas as pd
import numpy as np
from . import properties


class Loader():
    DATA_TYPE_USER = 'USER'
    DATA_TYPE_DEBATE = 'DEBATE'
    def __init__(self):
        self.users_df = self.get_users()
        self.debates_df = self.get_debates()
        return

    def get_users(self):
        users_df = pd.read_json(properties.DEBATEORG_USERS_JSON_PATH, orient='index')
        users_df.index.name= 'user_name'
        return users_df

    def get_debates(self):
        debates_df = pd.read_json(properties.DEBATEORG_DEBATES_JSON_PATH, orient='index')
        return debates_df

    def get_user_political_parties(self, plot = True):
        parties_count_df = self.get_value_counts_df('party',
                                                    data_type = Loader.DATA_TYPE_USER,
                                                    count_name='user_count',
                                                    index_name='political_party',
                                                    plot=plot)
        return parties_count_df

    def get_user_ideologies(self, plot = True):
        ideologies_count_df = self.get_value_counts_df('political_ideology',
                                                    data_type = Loader.DATA_TYPE_USER,
                                                    count_name='user_count',
                                                    index_name='ideology',
                                                    plot=plot)
        return ideologies_count_df

    def get_debate_categories(self, plot = True):
        categories_count_df  = self.get_value_counts_df('category',
                                                        data_type = Loader.DATA_TYPE_DEBATE,
                                                        count_name='debate_count',
                                                        index_name='debate_category',
                                                        plot=plot)
        return categories_count_df


    def get_value_counts_df(self, column, data_type, count_name= None, index_name = None, plot=False ):
        count_df = self.users_df[column].value_counts().to_frame(name=count_name) if data_type == Loader.DATA_TYPE_USER \
            else self.debates_df[column].value_counts().to_frame(name=count_name)
        count_df.index.name = index_name
        if plot:
            top = 15 if len(count_df) >15 else len(count_df)
            count_df.head(top).plot(kind='bar')
        return count_df