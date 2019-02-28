from user_based_recsys import UserBasedRecommender

from deep_implicit_feedback_recsys import *

class ColdStartImplicitRecommender(object):

    def __init__(self, nb_users, nb_items, state_history, action_history, reward_history,
            ImplicitRecommender=ImplicitRecommenderSimple,
            user_dim=32, item_dim=64,
            n_hidden=1, hidden_size=128, dropout=0.1, l2_reg=0,
            n_epochs=15, batch_size=64, online_batch_size=0, verbose=1):

        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._online_batch_size = online_batch_size
        self._verbose = verbose
        self._state_history = []
        self._action_history = []
        self._reward_history = []
        # Create and train models
        self._user_based_reco = UserBasedRecommender(state_history, action_history,
                reward_history)
        self._implicit_reco   = ImplicitRecommender(nb_users, nb_items, 
                user_dim=user_dim, item_dim=item_dim, n_hidden=n_hidden,
                hidden_size=hidden_size, dropout=dropout, l2_reg=l2_reg)
        self._implicit_reco.train(state_history, action_history, reward_history,
            n_epochs=n_epochs, batch_size=batch_size, verbose=verbose)


    def predict(self, state):
        user_id = state[0][0]
        prediction = None
        if self._user_based_reco.has_seen_user(user_id):
            prediction = self._implicit_reco.predict(state)
        else:
            prediction = self._user_based_reco.predict(state)
        return prediction


    def actualize(self, state, action, reward):
        self._user_based_reco.actualize(state, action, reward)
        if self._online_batch_size > 0:
            self._state_history.append(state)
            self._state_history.append(action)
            self._state_history.append(reward)
            if len(self._state_history) == self._batch_size:
                self._implicit_reco.train(self._state_history,
                        self._action_history, self._reward_history,
                        n_epochs=self._n_epochs, batch_size=self._batch_size,
                        verbose=self._verbose)
                self._state_history = []
                self._state_history = []
                self._state_history = []
        return self

class ColdStartFullyImplicitRecommender(object):
    def __init__(self, nb_users, nb_items, state_history, action_history, reward_history,
            ImplicitRecommender=ImplicitRecommenderSimple, user_dim=32, item_dim=64,
            n_hidden=1, hidden_size=128, dropout=0.1, l2_reg=0,
            n_epochs=15, batch_size=64, online_batch_size=0, 
            verbose=0):
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._online_batch_size = online_batch_size
        self._verbose = verbose
        self._state_history = []
        self._action_history = []
        self._reward_history = []
        self._user_based_reco = UserBasedRecommender(state_history, action_history, reward_history)
        self._implicit_reco   = ImplicitRecommenderSimple(nb_users, nb_items, 
                user_dim=user_dim, item_dim=item_dim, n_hidden=n_hidden, 
                hidden_size=hidden_size, dropout=dropout, l2_reg=l2_reg)

    def __str__(self):
        return 'ColdStartFullyImplicitRecommender'

    def predict(self, state):
        user_id = state[0][0]
        pred = None
        if not self._user_based_reco.has_seen_user(user_id):
            # Find similar user and modify the state
            # filter bought items with items available
            available_items = [n[1] for n in state]
            df_bought = self._user_based_reco._df_bought
            user_df   = self._user_based_reco._user_df
            filtered_bought_df = df_bought[df_bought.item_id.isin(available_items)]
            filtered_user_df   = user_df\
                    [user_df.user_id.isin(filtered_bought_df.user_id)]
            # find most similar user in the filtered df
            user_metadata = state[0][3:5]
            similar_user_id = self._user_based_reco.\
                    _similar_user(filtered_user_df, user_metadata)
            # modify the state with wimilar user found
            state[:][0] = similar_user_id
        pred = self._user_based_reco.predict(state)
        return pred

    def actualize(self, state, action, reward):
        self._user_based_reco.actualize(state, action, reward)
        if self._online_batch_size > 0:
            self._state_history.append(state)
            self._state_history.append(action)
            self._state_history.append(reward)
            if len(self._state_history) == self._batch_size:
                self._implicit_reco.train(self._state_history,
                        self._action_history, self._reward_history,
                        n_epochs=self._n_epochs, batch_size=self._batch_size,
                        verbose=self._verbose)
                self._state_history = []
                self._state_history = []
                self._state_history = []
        return self

