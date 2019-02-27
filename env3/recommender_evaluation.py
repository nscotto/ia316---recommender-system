import numpy as np

import requests

from cold_star_recommender import *


def get_input(user_id='VI2X71V0287S9F9B7SCU'):
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

def send_pred(pred, user_id='VI2X71V0287S9F9B7SCU'):
    adress = 'http://35.180.178.243'
    req = requests.get(adress + '/predict?user_id=' + user_id + '&recommended_item=' + str(pred))

    data = req.json()
    reward = data.pop('reward')
    state = data.pop('state')
    return reward, state


verbose = True
def inform(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def eval_loop_UserBasedRecommender(RecommenderClass, n_epochs=15, n_loop=10, n_pred=1000,
        online_batch_size=None, verbose=0, ID='VI2X71V0287S9F9B7SCU'):
    """ Evaluate a recommender erformance 
    RecommenderClass: Class of a recommender
        should be instantiable with RecommenderClass(nb_users, nb_items)
    n_loop: number of loop to mean the obtaimed result
    n_pred: number of prediction in each loop
    online_batch_size: if None, no online batch, 
        else a non-null integer equals to the number of predictions before
        actualizing the model (1 is for streaming)
    """
    rewards, rates = [], []
    for _ in range(n_loop):
        reward, rate = eval_UserBasedRecommender(RecommenderClass, n_epochs=n_epochs,
                n_pred=n_pred, online_batch_size=online_batch_size,
                verbose=verbose, ID=ID)
        rewards.append(reward)
        rates.append(rate)
    return rewards, rates

def eval_UserBasedRecommender(RecommenderClass, n_epochs=15, n_pred=1000,
        online_batch_size=None, verbose=0, ID='VI2X71V0287S9F9B7SCU'):
    """ Evaluate a recommender erformance 
    RecommenderClass: Class of a recommender
        should be instantiable with RecommenderClass(nb_users, nb_items)
    n_pred: number of predictions
    online_batch_size: if None, no online batch, 
        else a non-null integer equals to the number of predictions before
        actualizing the model (1 is for streaming)
    """

    nb_users, nb_items, next_state, data_array = get_input(ID)
    state_history, action_history, reward_history = data_array
    recommender = RecommenderClass(state_history, action_history, reward_history)
    rewards, pos_rewards = 0, 0
    if online_batch_size is None:  # no online training
        for _ in range(n_pred):
            pred = recommender.predict(next_state)
            reward, next_state = send_pred(pred, ID)
            rewards += reward
            pos_rewards += reward > 0
    else:
        state_history, action_history, reward_history = ([] for _ in range(3))
        for _ in range(n_pred):
            if len(state_history) == online_batch_size:  # retrain
                state_history, action_history, reward_history = (np.array(x) for x in 
                        (state_history, action_history, reward_history))
                recommender.train(state_history, action_history, reward_history,
                        verbose=verbose)
                state_history, action_history, reward_history = ([] for _ in range(3))
            pred = recommender.predict(next_state)
            state_history.append(next_state)
            action_history.append(pred)
            reward, next_state = send_pred(pred, ID)
            reward_history.append(reward)
            rewards += reward
            pos_rewards += reward > 0
    mean = rewards / n_pred
    rate = pos_rewards / n_pred
    return mean, rate

def eval_loop_ImplicitRecommender(RecommenderClass, n_epochs=15, n_loop=10, n_pred=1000,
        online_batch_size=None, verbose=0, ID='VI2X71V0287S9F9B7SCU'):
    """ Evaluate a recommender erformance 
    RecommenderClass: Class of a recommender
        should be instantiable with RecommenderClass(nb_users, nb_items)
    n_loop: number of loop to mean the obtaimed result
    n_pred: number of prediction in each loop
    online_batch_size: if None, no online batch, 
        else a non-null integer equals to the number of predictions before
        actualizing the model (1 is for streaming)
    """
    rewards, rates = [], []
    for _ in range(n_loop):
        reward, rate = eval_ImplicitRecommender(RecommenderClass, n_epochs=n_epochs,
                n_pred=n_pred, online_batch_size=online_batch_size,
                verbose=verbose, ID=ID)
        rewards.append(reward)
        rates.append(rate)
    return rewards, rates

def eval_ImplicitRecommender(RecommenderClass, n_epochs=15, n_pred=1000,
        verbose=0, ID='VI2X71V0287S9F9B7SCU'):
    """ Evaluate a recommender erformance 
    RecommenderClass: Class of a recommender
        should be instantiable with RecommenderClass(nb_users, nb_items)
    n_pred: number of predictions
    online_batch_size: if None, no online batch, 
        else a non-null integer equals to the number of predictions before
        actualizing the model (1 is for streaming)
    """

    nb_users, nb_items, next_state, data_array = get_input(ID)
    state_history, action_history, reward_history = data_array
    recommender = RecommenderClass(nb_users, nb_items)
    recommender.train(state_history, action_history, reward_history, 
            n_epochs=n_epochs, verbose=verbose)
    rewards, pos_rewards = 0, 0
    for _ in range(n_pred):
        pred = recommender.predict(next_state)
        next_state_bckp = next_state
        reward, next_state = send_pred(pred, ID)
        rewards += reward
        if reward > 0:
            pos_rewards += 1
            recommender.actualize(next_state_bckp)
    mean = rewards / n_pred
    rate = pos_rewards / n_pred
    return mean, rate


def eval_loop_ColdStartImplicitRecommender(ImplicitRecommenderClass, n_epochs=15, n_loop=10, n_pred=1000,
        online_batch_size=None, verbose=0, ID='VI2X71V0287S9F9B7SCU'):
    """ Evaluate a recommender erformance 
    RecommenderClass: Class of a recommender
        should be instantiable with RecommenderClass(nb_users, nb_items)
    n_loop: number of loop to mean the obtaimed result
    n_pred: number of prediction in each loop
    online_batch_size: if None, no online batch, 
        else a non-null integer equals to the number of predictions before
        actualizing the model (1 is for streaming)
    """
    rewards, rates = [], []
    for _ in range(n_loop):
        reward, rate = eval_ColdStartImplicitRecommender(ImplicitRecommenderClass, n_epochs=n_epochs,
                n_pred=n_pred, verbose=verbose, ID=ID)
        rewards.append(reward)
        rates.append(rate)
    return rewards, rates

def eval_ColdStartImplicitRecommender(ImplicitRecommenderClass, n_epochs=15, n_pred=1000,
        verbose=0, ID='VI2X71V0287S9F9B7SCU'):
    """ Evaluate a recommender erformance 
    RecommenderClass: Class of a recommender
        should be instantiable with RecommenderClass(nb_users, nb_items)
    n_pred: number of predictions
    online_batch_size: if None, no online batch, 
        else a non-null integer equals to the number of predictions before
        actualizing the model (1 is for streaming)
    """

    nb_users, nb_items, next_state, data_array = get_input(ID)
    state_history, action_history, reward_history = data_array
    recommender = ColdStartImplicitRecommender(nb_users, nb_items, state_history, action_history,
            reward_history, ImplicitRecommender=ImplicitRecommenderClass, n_epochs=n_epochs, verbose=verbose)
    rewards, pos_rewards = 0, 0
    for _ in range(n_pred):
        pred = recommender.predict(next_state)
        next_state_bckp = next_state
        reward, next_state = send_pred(pred, ID)
        rewards += reward
        if reward > 0:
            pos_rewards += 1
            recommender.actualize(next_state_bckp, pred, reward)
    mean = rewards / n_pred
    rate = pos_rewards / n_pred
    return mean, rate

