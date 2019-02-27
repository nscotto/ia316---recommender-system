import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Embedding, Flatten, Input, Dense, Dropout
from keras.layers import Concatenate, Lambda
from keras.regularizers import l2

def identity_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

def margin_comparator_loss(inputs, margin=1.):
    positive_pair_sim, negative_pair_sim = inputs
    return tf.maximum(negative_pair_sim - positive_pair_sim + margin, 0)


def make_interaction_mlp(input_dim, n_hidden=1, hidden_size=64,
                         dropout=0, last_activation=None, l2_reg=None):
    """Build the shared multi layer perceptron"""
    mlp = Sequential()
    if n_hidden == 0:
        # Plug the output unit directly: this is a simple
        # linear regression model. Not dropout required.
        mlp.add(Dense(1, input_dim=input_dim,
                      activation=last_activation, kernel_regularizer=l2_reg))
    else:
        mlp.add(Dense(hidden_size, input_dim=input_dim,
                      activation='relu', kernel_regularizer=l2_reg))
        mlp.add(Dropout(dropout))
        for i in range(n_hidden - 1):
            mlp.add(Dense(hidden_size, activation='relu',
                          W_regularizer=l2_reg))
            mlp.add(Dropout(dropout))
        mlp.add(Dense(1, activation=last_activation, kernel_regularizer=l2_reg))
    return mlp


def build_models(n_users, n_items, user_dim=32, item_dim=64, metadata_dim=7,
                 n_hidden=1, hidden_size=64, dropout=0, l2_reg=0, last_activation=None):
    """Build models to train a deep triplet network"""
    user_input = Input((1,), name='user_input')
    positive_item_input = Input((1,), name='positive_item_input')
    negative_item_input = Input((1,), name='negative_item_input')
    positive_metadata_input = Input((metadata_dim,), name='positive_metadata_input')
    negative_metadata_input = Input((metadata_dim,), name='negative_metadata_input')

    l2_reg = None if l2_reg == 0 else l2(l2_reg)
    user_layer = Embedding(n_users, user_dim, input_length=1,
                           name='user_embedding', embeddings_regularizer=l2_reg)

    # The following embedding parameters will be shared to encode both
    # the positive and negative items.
    item_layer = Embedding(n_items, item_dim, input_length=1,
                           name="item_embedding", embeddings_regularizer=l2_reg)

    user_embedding = Flatten()(user_layer(user_input))
    positive_item_embedding = Flatten()(item_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_layer(negative_item_input))


    # Similarity computation between embeddings using a MLP similarity
    positive_embeddings_pair = Concatenate(name="positive_embeddings_pair")(
        [user_embedding, positive_item_embedding, positive_metadata_input])
    positive_embeddings_pair = Dropout(dropout)(positive_embeddings_pair)
    negative_embeddings_pair = Concatenate(name="negative_embeddings_pair")(
        [user_embedding, negative_item_embedding, negative_metadata_input])
    negative_embeddings_pair = Dropout(dropout)(negative_embeddings_pair)

    # Instanciate the shared similarity architecture
    interaction_layers = make_interaction_mlp(
        user_dim + item_dim + metadata_dim, n_hidden=n_hidden, hidden_size=hidden_size,
        dropout=dropout, l2_reg=l2_reg, last_activation=last_activation)

    positive_similarity = interaction_layers(positive_embeddings_pair)
    negative_similarity = interaction_layers(negative_embeddings_pair)

    # The triplet network model, only used for training
    triplet_loss = Lambda(margin_comparator_loss, output_shape=(1,),
                          name='comparator_loss')(
        [positive_similarity, negative_similarity])

    deep_triplet_model = Model(inputs=[user_input,
                                       positive_item_input,
                                       negative_item_input,
                                       positive_metadata_input,
                                       negative_metadata_input],
                               outputs=[triplet_loss])

    # The match-score model, only used at inference
    deep_match_model = Model(inputs=[user_input, positive_item_input, positive_metadata_input],
                             outputs=[positive_similarity])

    return deep_match_model, deep_triplet_model

class ImplicitRecommenderSimple(object):
    def __init__(self, nb_users, nb_items, user_dim=32, item_dim=64,
            n_hidden=1, hidden_size=128, dropout=0.1, l2_reg=0):
        self._hyper_parameters = dict(
                user_dim=user_dim,
                item_dim=item_dim,
                n_hidden=n_hidden,
                hidden_size=hidden_size,
                dropout=dropout,
                l2_reg=l2_reg
                )
        self._deep_match_model, self._deep_triplet_model = build_models(nb_users,
                nb_items, metadata_dim=6, **self._hyper_parameters)
        self._deep_triplet_model.compile(loss=identity_loss, optimizer='adam')

    def __str__(self):
        return 'ImplicitRecommenderSimple' 

    def _sample_triplets(self, pos_state_data, pos_action_data, random_seed=0):
        """ Triplet sampling
        pos_state_data: the history of positive state
        pos_action_data: the history of actions related to positive states
        random_seed: random_seed generator

        return: user_ids, pos_items, pos_metadata, neg_items, neg_metadata
        """
        rng = np.random.RandomState(random_seed)
        user_ids, pos_items, pos_metadata, neg_items, neg_metadata = ([] for _ in range(5))
        for i in range(len(pos_action_data)):
            state  = pos_state_data[i]
            action = pos_action_data[i]
            user_ids.append(state[action][0])
            pos_items.append(state[action][1])
            pos_metadata.append(np.array(state[action][2:]))
            # Pick negative state
            k = action
            while k == action:
                k = rng.randint(0, len(state))
            neg_items.append(state[k][1])
            neg_metadata.append(np.array(state[k][2:]))
        return [np.array(x) for x in (user_ids, pos_items, neg_items, pos_metadata, neg_metadata)]

    def train(self, state_history, action_history, reward_history, n_epochs=15,
            batch_size=64, verbose=1):
        pos_rewards = reward_history > 0
        pos_rewards_history = reward_history[pos_rewards]
        pos_state_history  = state_history[pos_rewards]
        pos_action_history = action_history[pos_rewards]
        fake_y = np.ones_like(pos_rewards_history)

        for i in range(n_epochs):
            # Sample new negatives to build different triplets at each epoch
            triplet_inputs = self._sample_triplets(pos_state_history, pos_action_history,
                    random_seed=i)
            # Fit the model incrementally by doing a single pass over the
            # sampled triplets.
            self._deep_triplet_model.fit(triplet_inputs, fake_y, shuffle=True,
                    batch_size=batch_size, epochs=1, verbose=verbose)
        return self

    def predict(self, state):
        user_id = [state[0][0] for _ in range(len(state))]
        items   = [state[i][1] for i in range(len(state))]
        metadata = [state[i][2:] for i in range(len(state))]
        inputs = [np.array(x) for x in (user_id, items, metadata)]
        pred_score = self._deep_match_model.predict(inputs)
        pred = np.argmax(pred_score)
        return pred


class ImplicitRecommenderWithNull_no_indicator(object):
    def __init__(self, nb_users, nb_items, user_dim=32, item_dim=64,
            n_hidden=1, hidden_size=128, dropout=0.1, l2_reg=0):
        self._hyper_parameters = dict(
                user_dim=user_dim,
                item_dim=item_dim,
                n_hidden=n_hidden,
                hidden_size=hidden_size,
                dropout=dropout,
                l2_reg=l2_reg
                )
        self._deep_match_model, self._deep_triplet_model = build_models(nb_users,
                nb_items, metadata_dim=6, **self._hyper_parameters)
        self._deep_triplet_model.compile(loss=identity_loss, optimizer='adam')

    def __str__(self):
        return 'ImplicitRecommenderWithNull_no_indicator' 

    def _sample_triplets(self, pos_state_data, pos_action_data, random_seed=0):
        """ Triplet sampling
        pos_state_data: the history of positive state
        pos_action_data: the history of actions related to positive states
        random_seed: random_seed generator

        return: user_ids, pos_items, pos_metadata, neg_items, neg_metadata
        """
        rng = np.random.RandomState(random_seed)
        user_ids, pos_items, pos_metadata, neg_items, neg_metadata = ([] for _ in range(5))
        for i in range(len(pos_action_data)):
            state  = pos_state_data[i]
            action = pos_action_data[i]
            user_ids.append(state[action][0])
            pos_items.append(state[action][1])
            pos_metadata.append(np.array(state[action][2:]))
            # Pick negative state
            k = action
            while k == action:
                k = rng.randint(0, len(state))
            neg_items.append(state[k][1])
            neg_metadata.append(np.array(state[k][2:]))
        return [np.array(x) for x in (user_ids, pos_items, neg_items, pos_metadata, neg_metadata)]

    def _sample_triplets_null_reward(self, state_data, action_data, random_seed=0):
        """ Triplet sampling
        pos_state_data: the history of positive state
        pos_action_data: the history of actions related to positive states
        random_seed: random_seed generator

        return: user_ids, pos_items, pos_metadata, neg_items, neg_metadata
        """
        rng = np.random.RandomState(random_seed)
        user_ids, pos_items, pos_metadata, neg_items, neg_metadata = ([] for _ in range(5))
        for i in range(len(action_data)):
            state  = state_data[i]
            action = action_data[i]
            user_ids.append(state[action][0])
            neg_items.append(state[action][1])
            neg_metadata.append(np.array(state[action][2:]))
            # Pick negative state
            k = action
            while k == action:
                k = rng.randint(0, len(state))
            pos_items.append(state[k][1])
            pos_metadata.append(np.array(state[k][2:]))
        return [np.array(x) for x in (user_ids, pos_items, neg_items, pos_metadata, neg_metadata)]

    def train(self, state_history, action_history, reward_history, n_epochs=15,
            batch_size=64, verbose=1):
        pos_rewards = reward_history > 0
        pos_state_history  = state_history[pos_rewards]
        pos_action_history = action_history[pos_rewards]
        null_rewards = reward_history == 0
        null_state_history  = state_history[null_rewards]
        null_action_history = action_history[null_rewards]
        fake_y = np.ones_like(reward_history)

        for i in range(n_epochs):
            # Sample new negatives to build different triplets at each epoch
            triplet = self._sample_triplets(pos_state_history, pos_action_history,
                    random_seed=i)
            null_triplet = self._sample_triplets_null_reward(null_state_history,
                    null_action_history, random_seed=i)
            triplet_inputs = [np.concatenate((triplet[i], null_triplet[i]))
                    for i in range(len(triplet))]

            # Fit the model incrementally by doing a single pass over the
            # sampled triplets.
            self._deep_triplet_model.fit(triplet_inputs, fake_y, shuffle=True,
                    batch_size=batch_size, epochs=1, verbose=verbose)
        return self

    def predict(self, state):
        user_id = [state[0][0] for _ in range(len(state))]
        items   = [state[i][1] for i in range(len(state))]
        metadata = [state[i][2:] for i in range(len(state))]
        inputs = [np.array(x) for x in (user_id, items, metadata)]
        pred_score = self._deep_match_model.predict(inputs)
        pred = np.argmax(pred_score)
        return pred

class ImplicitRecommenderWithNull_binary_indicator(object):
    def __init__(self, nb_users, nb_items, user_dim=32, item_dim=64,
            n_hidden=1, hidden_size=128, dropout=0.1, l2_reg=0):
        self._hyper_parameters = dict(
                user_dim=user_dim,
                item_dim=item_dim,
                n_hidden=n_hidden,
                hidden_size=hidden_size,
                dropout=dropout,
                l2_reg=l2_reg
                )
        self._deep_match_model, self._deep_triplet_model = build_models(nb_users,
                nb_items, **self._hyper_parameters)
        self._deep_triplet_model.compile(loss=identity_loss, optimizer='adam')

    def __str__(self):
        return 'ImplicitRecommenderWithNull_binary_indicator' 

    def _sample_triplets(self, pos_state_data, pos_action_data, random_seed=0):
        """ Triplet sampling
        pos_state_data: the history of positive state
        pos_action_data: the history of actions related to positive states
        random_seed: random_seed generator

        return: user_ids, pos_items, pos_metadata, neg_items, neg_metadata
        """
        rng = np.random.RandomState(random_seed)
        user_ids, pos_items, pos_metadata, neg_items, neg_metadata = ([] for _ in range(5))
        for i in range(len(pos_action_data)):
            state  = pos_state_data[i]
            action = pos_action_data[i]
            user_ids.append(state[action][0])
            pos_items.append(state[action][1])
            pos_metadata.append(np.concatenate((state[action][2:], [0])))
            # Pick negative state
            k = action
            while k == action:
                k = rng.randint(0, len(state))
            neg_items.append(state[k][1])
            neg_metadata.append(np.concatenate((state[k][2:], [0])))
        return [np.array(x) for x in (user_ids, pos_items, neg_items, pos_metadata, neg_metadata)]

    def _sample_triplets_null_reward(self, state_data, action_data, random_seed=0):
        """ Triplet sampling
        pos_state_data: the history of positive state
        pos_action_data: the history of actions related to positive states
        random_seed: random_seed generator

        return: user_ids, pos_items, pos_metadata, neg_items, neg_metadata
        """
        rng = np.random.RandomState(random_seed)
        user_ids, pos_items, pos_metadata, neg_items, neg_metadata = ([] for _ in range(5))
        for i in range(len(action_data)):
            state  = state_data[i]
            action = action_data[i]
            user_ids.append(state[action][0])
            neg_items.append(state[action][1])
            neg_metadata.append(np.concatenate((state[action][2:], [1])))
            # Pick negative state
            k = action
            while k == action:
                k = rng.randint(0, len(state))
            pos_items.append(state[k][1])
            pos_metadata.append(np.concatenate((state[k][2:], [1])))
        return [np.array(x) for x in (user_ids, pos_items, neg_items, pos_metadata, neg_metadata)]

    def train(self, state_history, action_history, reward_history, n_epochs=15,
            batch_size=64, verbose=1):
        pos_rewards = reward_history > 0
        pos_state_history  = state_history[pos_rewards]
        pos_action_history = action_history[pos_rewards]
        null_rewards = reward_history == 0
        null_state_history  = state_history[null_rewards]
        null_action_history = action_history[null_rewards]
        fake_y = np.ones_like(reward_history)

        for i in range(n_epochs):
            # Sample new negatives to build different triplets at each epoch
            triplet = self._sample_triplets(pos_state_history, pos_action_history,
                    random_seed=i)
            null_triplet = self._sample_triplets_null_reward(null_state_history,
                    null_action_history, random_seed=i)
            triplet_inputs = [np.concatenate((triplet[i], null_triplet[i]))
                    for i in range(len(triplet))]

            # Fit the model incrementally by doing a single pass over the
            # sampled triplets.
            self._deep_triplet_model.fit(triplet_inputs, fake_y, shuffle=True,
                    batch_size=batch_size, epochs=1, verbose=verbose)
        return self

    def predict(self, state):
        user_id = [state[0][0] for _ in range(len(state))]
        items   = [state[i][1] for i in range(len(state))]
        metadata = [state[i][2:] + [0] for i in range(len(state))]
        inputs = [np.array(x) for x in (user_id, items, metadata)]
        pred_score = self._deep_match_model.predict(inputs)
        pred = np.argmax(pred_score)
        return pred


class ImplicitRecommenderWithNull(object):
    def __init__(self, nb_users, nb_items, user_dim=32, item_dim=64,
            n_hidden=1, hidden_size=128, dropout=0.1, l2_reg=0):
        self._hyper_parameters = dict(
                user_dim=user_dim,
                item_dim=item_dim,
                n_hidden=n_hidden,
                hidden_size=hidden_size,
                dropout=dropout,
                l2_reg=l2_reg
                )
        self._deep_match_model, self._deep_triplet_model = build_models(nb_users,
                nb_items, **self._hyper_parameters)
        self._deep_triplet_model.compile(loss=identity_loss, optimizer='adam')

    def __str__(self):
        return 'ImplicitRecommenderWithNull' 

    def _sample_triplets(self, pos_state_data, pos_action_data, random_seed=0):
        """ Triplet sampling
        pos_state_data: the history of positive state
        pos_action_data: the history of actions related to positive states
        random_seed: random_seed generator

        return: user_ids, pos_items, pos_metadata, neg_items, neg_metadata
        """
        rng = np.random.RandomState(random_seed)
        user_ids, pos_items, pos_metadata, neg_items, neg_metadata = ([] for _ in range(5))
        for i in range(len(pos_action_data)):
            state  = pos_state_data[i]
            action = pos_action_data[i]
            user_ids.append(state[action][0])
            pos_items.append(state[action][1])
            pos_metadata.append(np.concatenate((state[action][2:], [state[action][2]])))
            # Pick negative state
            k = action
            while k == action:
                k = rng.randint(0, len(state))
            neg_items.append(state[k][1])
            neg_metadata.append(np.concatenate((state[k][2:], [0])))
        return [np.array(x) for x in (user_ids, pos_items, neg_items, pos_metadata, neg_metadata)]

    def _sample_triplets_null_reward(self, state_data, action_data, random_seed=0):
        """ Triplet sampling
        pos_state_data: the history of positive state
        pos_action_data: the history of actions related to positive states
        random_seed: random_seed generator

        return: user_ids, pos_items, pos_metadata, neg_items, neg_metadata
        """
        rng = np.random.RandomState(random_seed)
        user_ids, pos_items, pos_metadata, neg_items, neg_metadata = ([] for _ in range(5))
        for i in range(len(action_data)):
            state  = state_data[i]
            action = action_data[i]
            user_ids.append(state[action][0])
            neg_items.append(state[action][1])
            neg_metadata.append(np.concatenate((state[action][2:], [-state[action][2]])))
            # Pick negative state
            k = action
            while k == action:
                k = rng.randint(0, len(state))
            pos_items.append(state[k][1])
            pos_metadata.append(np.concatenate((state[k][2:], [0])))
        return [np.array(x) for x in (user_ids, pos_items, neg_items, pos_metadata, neg_metadata)]

    def train(self, state_history, action_history, reward_history, n_epochs=15,
            batch_size=64, verbose=1):
        pos_rewards = reward_history > 0
        pos_state_history  = state_history[pos_rewards]
        pos_action_history = action_history[pos_rewards]
        null_rewards = reward_history == 0
        null_state_history  = state_history[null_rewards]
        null_action_history = action_history[null_rewards]
        fake_y = np.ones_like(reward_history)
        for i in range(n_epochs):
            # Sample new negatives to build different triplets at each epoch
            triplet_inputs = None
            if len(pos_state_history) > 0 and len(null_state_history) > 0:
                triplet = self._sample_triplets(pos_state_history, pos_action_history,
                        random_seed=i)
                null_triplet = self._sample_triplets_null_reward(null_state_history,
                        null_action_history, random_seed=i)
                triplet_inputs = [np.concatenate((triplet[i], null_triplet[i]))
                    for i in range(5)]
            elif len(pos_state_history) > 0:
                triplet_inputs = self._sample_triplets(pos_state_history, pos_action_history,
                        random_seed=i)
            elif len(null_state_history) > 0:
                triplet_inputs = self._sample_triplets_null_reward(null_state_history,
                        null_action_history, random_seed=i)
            # Fit the model incrementally by doing a single pass over the
            # sampled triplets.
            self._deep_triplet_model.fit(triplet_inputs, fake_y, shuffle=True,
                    batch_size=batch_size, epochs=1, verbose=verbose)
        return self

    def predict(self, state):
        user_id = [state[0][0] for _ in range(len(state))]
        items   = [state[i][1] for i in range(len(state))]
        metadata = [state[i][2:] + [0] for i in range(len(state))]
        inputs = [np.array(x) for x in (user_id, items, metadata)]
        pred_score = self._deep_match_model.predict(inputs)
        pred = np.argmax(pred_score)
        return pred

