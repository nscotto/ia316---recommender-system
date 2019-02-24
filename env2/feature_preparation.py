import numpy as np
import pandas as pd
import requests


class FeatureGenerator(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.nb_items = None
        self.nb_users = None
        self.next_item = None
        self.next_user = None
        self.next_variables = None


    def reset(self):
        adress = 'http://35.180.254.42'
        req = requests.get(adress + '/reset?user_id=' + self.user_id)
        data = req.json()
        self.nb_items = data['nb_items']
        self.nb_users = data['nb_users']
        X_train = np.hstack((np.array(data['user_history']).reshape((-1,1)),
                                  np.array(data['item_history']).reshape((-1,1)),
                                  np.array(data['variables_history'])))
        columns = ['user_id', 'item_id'] + ['var' + str(i) for i in range(5)]
        self.X_train = pd.DataFrame(X_train, columns=columns)
        self.y_train = np.array(data['rating_history'])

        self.next_item = data['next_item']
        self.next_user = data['next_user']
        self.next_variables = data['next_variables']
