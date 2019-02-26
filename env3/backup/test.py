# -*- coding: utf-8 -*-
"""
TODO install docker  --> hello world
$ docker run hello-world

puis installer docker-compose

"""


import sys
import numpy as np

import requests

from deep_implicit_feedback_recsys import *

#%%

def get_input(user_id):
    adress = 'http://35.180.178.243'

    req = requests.get(adress + '/reset?user_id=' + user_id)

    data = req.json()

    nb_users = data.pop('nb_users')
    nb_items = data.pop('nb_items')
    next_state = data.pop('next_state')
    columns = ['state_history', 'action_history', 'rewards_history']
    data_array = [np.array(data.pop(c)) for c in columns]
    """
    data_array = np.array([data.pop[c] for c in columns])
    df =  pd.DataFrame(data_array.T, columns=columns)
    """
    return nb_users, nb_items, next_state, data_array


verbose = True
def inform(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)



#%%
if __name__ == '__main__':
    ID = 'VI2X71V0287S9F9B7SCU'
    inform('Requesting history...')
    nb_users, nb_items, next_state, data_array = get_input(ID)
    state_history, action_history, reward_history = data_array
    inform('history retrieved')
    inform('creating model')
    hyper_parameters = dict(
        user_dim=32,
        item_dim=64,
        n_hidden=1,
        hidden_size=128,
        dropout=0.1,
        l2_reg=0
    )
    deep_match_model, deep_triplet_model = build_models(nb_users, nb_items,
                                                        **hyper_parameters)


    deep_triplet_model.compile(loss=identity_loss, optimizer='adam')

    pos_rewards = reward_history > 0
    pos_reward_history = reward_history[pos_rewards]
    pos_state_history  = state_history[pos_rewards]
    pos_action_history = action_history[pos_rewards]
    fake_y = np.ones_like(pos_reward_history)

    n_epochs = 15
    if len(sys.argv) > 1:
        n_epochs = int(sys.argv[1])

    for i in range(n_epochs):
        # Sample new negatives to build different triplets at each epoch
        triplet_inputs = sample_triplets(pos_state_history, pos_action_history,
                                         random_seed=i)

        # Fit the model incrementally by doing a single pass over the
        # sampled triplets.
        deep_triplet_model.fit(triplet_inputs, fake_y, shuffle=True,
                               batch_size=64, epochs=1)

        # Monitor the convergence of the model
    inform('Model fitted')
    inform('Predicting...')
    next_user_id = [next_state[0][0] for _ in range(len(next_state))]
    next_items   = [next_state[i][1] for i in range(len(next_state))]
    next_metadata = [next_state[i][2:] for i in range(len(next_state))]
    inputs = [np.array(x) for x in (next_user_id, next_items, next_metadata)]
    pred = deep_match_model.predict(inputs)
    print(pred)
