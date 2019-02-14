# -*- coding: utf-8 -*-
"""
TODO install docker  --> hello world
$ docker run hello-world

puis installer docker-compose

"""


from time import sleep
import pandas as pd
import numpy as np

import requests
from collab_filtering import df_to_ratings, predict_ratings_bias_sub



#%%

def get_input(user_id):
    adress = 'http://35.180.178.243'

    req = requests.get(adress + '/reset?user_id=' + user_id)

    data = req.json()

    nb_users = data.pop('nb_users')
    nb_items = data.pop('nb_items')
    next_state = data.pop('next_state')
    columns = ['state_history', 'reward_history', 'item_history', 'action_history']
    data_array = np.array([data.pop[c] for c in columns])
    df =  pd.DataFrame(data_array.T, columns=columns)

    return nb_users, nb_items, next_state, df

#%%

def predict_value(next_state, model=None):
    # input prediction model: score between 1 and 5

    #collab filtering with bias
    pred = model[next_state]

    return pred

#%%

def predict_all(user_id, early_stop, nb_samples):
    adress = 'http://35.180.254.42' #'http://35.180.46.68'

    df, dic = get_input(user_id)

    n_users, n_items = dic['nb_users'], dic['nb_items']

    next_u, next_i = dic['next_user'], dic['next_item']

    ratings = df_to_ratings(df)

    # collab filter with bias
    pred_model = predict_ratings_bias_sub(ratings)    
    pred = predict_value(next_u, next_i, pred_model)

    mse, mae = 0, 0
    i = 0.

    #for i in range(99):
    while i < nb_samples:
        i +=1.
        sleep(0.05)
        req = requests.get(adress + '/predict', 
                params={'user_id': user_id, 'predicted_score': pred})
        #catch error 404 when all database is parsed

        data = req.json()

        true_rating  = data['rating']
        next_u, next_i = data['next_user'], data['next_item']

        mse += (pred - true_rating) ** 2
        mae += abs(pred - true_rating)
        print('iteration {} SE={:.3f}'.format(i, (pred-true_rating)**2))        
        pred = predict_value(next_u, next_i, pred_model)

    print('Done \n MSE={:.3f} \n MAE={:.3f}'.format(mse/i, mae/i))

    return mse/i, mae/i












#%%
"""
params={'user_id': userid}

data = req.json()

nb_users, ..., next_item = data.pop(..)... , data.pop('next_item')


prediction = 



for i in range(nb_samples):
    sleep(0.05)
    r = requests.get(url_predit, params=params)
    d = r.json()

    rating = d['rating']

    mase += (rating - prediction)**2


"""

#%%
