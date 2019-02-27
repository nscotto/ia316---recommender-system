import pandas as pd

class UserBasedRecommender(object):
    def __init__(nb_users, state_history, action_history, reward_history):
        user_data = state_history[:,0,
        self._user_df = pd.DataFrame(np.arange(nb_users), colums='user_id').set_index('user_id')


