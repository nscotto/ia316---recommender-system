import pandas as pd

import numpy as np
from numpy import linalg

class UserBasedRecommender(object):
    def __init__(self, state_history, action_history, reward_history):
        columns_names = ['user_id','item_id', 'item_price'] + ['meta_' + str(i)
                for i in range(5)]
        df = pd.DataFrame(state_history[0], columns=columns_names)
        df['hist_line'] = 0
        df['action_item_id'] = state_history[0][action_history[0]][1]
        df['reward'] = reward_history[0]
        for i in range(1, len(state_history)):
            df_temp = pd.DataFrame(state_history[i], columns=columns_names)
            df_temp['hist_line'] = i
            df_temp['action_item_id'] = state_history[i][action_history[i]][1]
            df_temp['reward'] = reward_history[i]
            df = df.append(df_temp, ignore_index=True, sort=False)

        # keep only informations with positive rewards
        df_bought = df[df.reward > 0]
        df_bought = df_bought[df_bought.action_item_id == df_bought.item_id]
        user_df = df_bought.drop_duplicates('user_id')\
                .sort_values('user_id')[['user_id', 'meta_0', 'meta_1']]
        # normalize meta data
        norm = linalg.norm(user_df[['meta_0', 'meta_1']].values, axis=1)
        user_df['meta_0'] = user_df.meta_0.values / norm
        user_df['meta_1'] = user_df.meta_1.values / norm
        self._df_bought = df_bought
        self._user_df = user_df
        self._columns_names = columns_names

    def _similar_user(self, df, user_metadata):
        # normalize user metadata
        user_metadata = np.array(user_metadata) / linalg.norm(user_metadata)
        # compute cosine similarity
        similarity = df[['meta_0', 'meta_1']].values.dot(user_metadata)
        similar_user_id = df.user_id.iloc[np.argmax(similarity)]
        return similar_user_id

    def predict(self, next_state):
        # filter bought items with items available
        available_items = [n[1] for n in next_state]
        filtered_bought_df = self._df_bought[self._df_bought.item_id.isin(available_items)]
        filtered_user_df   = self._user_df\
                [self._user_df.user_id.isin(filtered_bought_df.user_id)]
        # find most similar user in the filtered df
        user_metadata = next_state[0][3:5]
        similar_user_id = self._similar_user(filtered_user_df, user_metadata)
        items = filtered_bought_df[filtered_bought_df.user_id == similar_user_id]
        items = items.sort_values('item_price', ascending=False)
        predict_item_id = items.item_id.iloc[0]
        prediction = available_items.index(predict_item_id)
        return prediction

    def actualize(self, state, action, reward):
        if reward > 0:
            df = pd.DataFrame([state[action]], columns=self._columns_names)
            self._df_bought = self._df_bought.append(df, ignore_index=True, sort=False)
            user_id = df.user_id.iloc[0]
            if not self.has_seen_user(user_id):
                # add user and its metadata to user df
                # normalize meta data
                norm = linalg.norm(df[['meta_0', 'meta_1']].values, axis=1)
                df['meta_0'] = df.meta_0.values / norm
                df['meta_1'] = df.meta_1.values / norm
                self._user_df = self._user_df.append(df, ignore_index=True, sort=False)
        return self

    def has_seen_user(self, user_id):
        return user_id in self._user_df.user_id

