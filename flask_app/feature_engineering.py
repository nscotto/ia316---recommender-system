import json
import numpy as np
import requests



class FeatureGenerator(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.nb_users = None
        self.nb_items = None
        self.action_history = None
        self.rewards_history = None
        self.state_history = None
        self.next_state = None

    def reset(self):
        adress = 'http://35.180.178.243'
        req = requests.get(adress + '/reset?user_id=' + self.user_id)
        data = req.json()

        self.nb_items = data['nb_items']
        self.nb_users = data['nb_users']

        self.action_history = data['action_history']
        self.state_history = data['state_history']
        self.rewards_history = data['rewards_history']
        return self

    def parse_data(self):
        """ parses the history, returns history as (state,  rewards)

        Returns
        -------

        """
        res = []
        for state, action, reward in zip(self.state_history,
                                        self.action_history,
                                        self.rewards_history):
            for state_j in state: # iterates available items
                if state_j[1] == action:
                    res.append((state_j, action, reward))
                else:
                    res.append((state_j, state_j[1], state_j[2]))
        res = np.array(res).reshape((len(res), 3))
        mask_pos = np.where(res[:, 2] > 0, True, False)

        res_pos = res[mask_pos]
        res_neg = res[~mask_pos]
        return res_pos, res_neg

# main for debugging
# if __name__ == '__main__':
#     gen = FeatureGenerator('MC15CHD1JU24INEKKPU4')
#     gen.reset()
#     res = gen.parse_data()
#     print('done')
