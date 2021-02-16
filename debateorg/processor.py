import json
import pandas as pd
import numpy as np
import glob
from . import properties
from . import utils
from . import loader
import os
import collections
#         Before              Yes             No
# After
#      Yes                   OKAY            Effective
#       No            Provocative          Ineffective
class Process():
    def __init__(self):
        self.vote_w_effect_df = self.get_votes_w_effect()
        self.loader = loader.Loader()
        self.loader.get_users()
        self.loader.get_debates()


    ##########################################################
    # HELPERS
    ##########################################################
    @staticmethod
    def _debate_effect(before, after):
        effect = ''
        if before and after:
            effect = 'okay'
        elif before and not after:
            effect = 'provocative'
        elif not before and not after:
            effect = 'ineffective'
        elif not before and after:
            effect = 'effective'

        return effect

    @staticmethod
    def _add_effect(row):
        # check voter and p1
        row['p1_effect'] = Process._debate_effect(row['p1_agree_before'], row['p1_agree_after'])
        row['p2_effect'] = Process._debate_effect(row['p2_agree_before'], row['p2_agree_after'])
        combined_effect = [row['p1_effect'], row['p2_effect']]
        combined_effect.sort()
        row['effect'] = '-'.join(combined_effect)
        return row

    @staticmethod
    def _add_opposite_interaction(row):
        interactions = []
        ideology = row['voter_ideology']
        if row['p1_ideology'] != ideology:
            interactions.append('p1')
        if row['p2_ideology'] != ideology:
            interactions.append('p2')
        row['interaction'] = ','.join(interactions)
        return row

    # get voter participant row for each vote where voter ideology is different than the participant's
    def get_voter_particpant_df(self, df): #get_liberal_conservative_flat
        # each row should be voter with a participant
        result = []
        for i, row in df.iterrows():

            info = {}
            info['debate_id'] = row['debate_id']
            info['category'] = row['category']

            info['voter_username'] = row['voter_username']
            info['voter_ideology'] = row['voter_ideology']

            for participant in row['interaction'].split(','):
                single_inter = info
                single_inter['p_ideology'] = row['{}_ideology'.format(participant)]

                single_inter['p_name'] = row['{}_name'.format(participant)]

                single_inter['p_agree_before'] = row['{}_agree_before'.format(participant)]
                single_inter['p_agree_after'] = row['{}_agree_after'.format(participant)]
                single_inter['p_convincing'] = row['{}_convincing'.format(participant)]
                single_inter['p_effect'] = row['{}_effect'.format(participant)]
                result.append(single_inter.copy())
        return pd.DataFrame(result)

    def get_debate_participant_arguments(self, debate_id, participant):
        arguments = []
        not_found_args = {}
        debate = self.loader.debates_df.loc[debate_id]

        # debate = debate[(debates['participant_1_name'] == participant) | \
        #                 (debates['participant_2_name'] == participant)]#['rounds']
        # print(debate)

        p_id = -1
        if debate['participant_1_name'] == participant:
            p_id = 1
        elif debate['participant_2_name'] == participant:
            p_id = 2
        else:
            print('ERROR NOT MATCHING ', debate_id)
            return arguments
        position = debate['participant_{}_position'.format(p_id)]

        rounds = debate['rounds']
        found = False
        for r in rounds:
            try:
                for elt in r:
                    if position == elt['side']:
                        arg = elt['text']
                        found = True
                        arguments.append(arg)

            except Exception as e:
                print()
                print(' EXCEPTION ')
                print(debate_id, ' ', participant)
        if not found:
            not_found_args = {
                'debate_id': debate_id,
                'participant': participant,
                'rounds': rounds
            }
        return arguments, not_found_args
    ##########################################################
    # END OF HELPERS
    ##########################################################


    def get_votes_w_effect(self):
        if os.path.isfile(properties.FLAT_VOTES_W_EFFECT_CSV):
            vote_w_effect_df = pd.read_csv(properties.FLAT_VOTES_W_EFFECT_CSV)
        else:

            vote_w_effect_df, dismissed = self.loader.flatten_debate_votes()
            vote_w_effect_df = vote_w_effect_df.apply(Process._add_effect, axis=1)

            vote_w_effect_df.to_csv(properties.FLAT_VOTES_W_EFFECT_CSV)
            print('dismissed:', len(dismissed))
        return vote_w_effect_df



    def get_args_w_main_effect_df(self, df, flatten=False): # get_debates_with_overlaping_effect
        result = []
        not_found_arr = []
        for debate_id, debate_df in df.groupby(['debate_id', 'p_name']):
            effects = debate_df['p_effect'].unique().tolist()
            effects.sort()

            effect_counter = dict(collections.Counter(debate_df['p_effect'].values.tolist()))

            maj_value = max(effect_counter.values())
            top_effect = [key for key, value in effect_counter.items() if value == maj_value]
            top_effect.sort()

            if 'effective' in effects:
                effect = 'effective'
            elif 'ineffective' in effects:
                effect = 'ineffective'
            elif 'provocative' in effects:
                effect = 'provocative'
            elif 'okay' in effects:
                effect = 'okay'
            else:
                effect = 'NA'

            arguments, not_found_dict = self.get_debate_participant_arguments(debate_id[0], debate_id[1])

            if len(arguments) == 0:
                not_found_arr.append(not_found_dict)
                continue

            debate = {
                'id': '|'.join(debate_id),
                'debate_id': debate_id[0],
                'p_name': debate_id[1],
                'effects': '-'.join(effects),
                'effect_count': effect_counter,
                'top_effect': '-'.join(top_effect),
                'effect': effect,
                'category': debate_df['category'].values[0]
            }

            if flatten:
                for i in range(len(arguments)):
                    arg = arguments[i].strip()
                    if arg != 'forfeit' and arg != '':
                        single_arg_entry = debate.copy()
                        single_arg_entry['round'] = i
                        single_arg_entry['argument'] = arguments[i]
                        result.append(single_arg_entry)


            else:
                debate['arguments'] = arguments
                result.append(debate)
        not_found_df = pd.DataFrame(not_found_arr)
        return pd.DataFrame(result), not_found_df

    # MAIN DATASET
    def get_ideology_based_voter_participant_df(self, ideology):
        file_path = properties.DEBATEORG_ARGS_W_EFFECT_PATH.format(ideology.lower())
        if os.path.isfile(file_path):
            arguments_dataset = pd.read_csv(file_path,
                                            index_col='numeric_id')
            not_found_args_df = pd.read_csv(properties.MISSING_ARGS_CSV.format(ideology.lower()))
        else:
            df = self.vote_w_effect_df[(self.vote_w_effect_df['voter_ideology'] == ideology) & \
                                       ((self.vote_w_effect_df['p1_ideology'] != ideology) |
                                        (self.vote_w_effect_df['p2_ideology'] != ideology))]
            df = df.apply(Process._add_opposite_interaction, axis=1)
            dataset = self.get_voter_particpant_df(df)

            arguments_dataset, not_found_args_df = self.get_args_w_main_effect_df(dataset, flatten=True)
            arguments_dataset.reset_index().to_csv(file_path,
                                                   index=False)

            not_found_args_df.dropna(inplace=True)
            not_found_args_df.to_csv(properties.MISSING_ARGS_CSV.format(ideology.lower()))
        return arguments_dataset, not_found_args_df