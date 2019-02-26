import numpy as np

import requests


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

def send_pred(user_id, pred):
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


def eval_loop_recommender(RecommenderClass, n_epochs=15, n_loop=10, n_pred=1000,
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
    rewards, rates = []
    for _ in range(n_loop):
        reward, rate = eval_recommender(RecommenderClass, n_epochs=n_epochs,
                n_pred=n_pred, online_batch_size=online_batch_size,
                verbose=verbose, ID=ID)
        rewards.append(reward)
        rates.append(rate)
    return rewards, rates

def eval_recommender(RecommenderClass, n_epochs=15, n_pred=1000,
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
    recommender = RecommenderClass(nb_users, nb_items)
    recommender.train(state_history, action_history, reward_history, 
            n_epochs=n_epochs, verbose=verbose)
    rewards, pos_rewards = 0, 0
    if online_batch_size is None:  # no online training
        for _ in range(n_pred):
            pred = recommender.predict(next_state)
            reward, next_state = send_pred(ID, pred)
            rewards += reward
            pos_rewards += reward > 0
    else:
        state_history, action_history, reward_history = ([] for _ in range(3))
        for _ in range(n_pred):
            if len(state_history) == online_batch_size:  # retrain
                recommender.train(state_history, action_history, reward_history)
                state_history, action_history, reward_history = ([] for _ in range(3))
            pred = recommender.predict(next_state)
            reward, next_state = send_pred(ID, pred)
            state_history.append(next_state)
            reward_history.append(reward)
            action_history.append(pred)
            rewards += reward
            pos_rewards += reward > 0
    mean = rewards / n_pred
    rate = pos_rewards / n_pred
    return mean, rate
