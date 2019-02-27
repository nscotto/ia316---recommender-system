from user_based_recsys import UserBasedRecommender

from deep_implicit_feedback_recsys import *

class ColdStartImplicitRecommender(object):
    def __init__(self, nb_users, nb_items, state_history, action_history, reward_history,
            ImplicitRecommender=ImplicitRecommenderSimple,
            online_batch_size=1, n_epochs=15, batch_size=64, verbose=1):
        self._user_based_reco = UserBasedRecommender(state_history, action_history,
                reward_history)
        self._implicit_reco   = ImplicitRecommender(nb_users, nb_items)
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


