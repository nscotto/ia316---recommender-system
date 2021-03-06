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

def sample_triplets(pos_state_data, pos_action_data, random_seed=0):
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

def sample_triplets_null_reward(state_data, action_data, random_seed=0):
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


def make_interaction_mlp(input_dim, n_hidden=1, hidden_size=64,
                         dropout=0, l2_reg=None):
    """Build the shared multi layer perceptron"""
    mlp = Sequential()
    if n_hidden == 0:
        # Plug the output unit directly: this is a simple
        # linear regression model. Not dropout required.
        mlp.add(Dense(1, input_dim=input_dim,
                      activation=None, kernel_regularizer=l2_reg))
    else:
        mlp.add(Dense(hidden_size, input_dim=input_dim,
                      activation='relu', kernel_regularizer=l2_reg))
        mlp.add(Dropout(dropout))
        for i in range(n_hidden - 1):
            mlp.add(Dense(hidden_size, activation='relu',
                          W_regularizer=l2_reg))
            mlp.add(Dropout(dropout))
        mlp.add(Dense(1, activation=None, kernel_regularizer=l2_reg))
    return mlp


def build_models(n_users, n_items, user_dim=32, item_dim=64,
                 n_hidden=1, hidden_size=64, dropout=0, l2_reg=0):
    """Build models to train a deep triplet network"""
    user_input = Input((1,), name='user_input')
    positive_item_input = Input((1,), name='positive_item_input')
    negative_item_input = Input((1,), name='negative_item_input')
    positive_metadata_input = Input((6,), name='positive_metadata_input')
    negative_metadata_input = Input((6,), name='negative_metadata_input')

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
        user_dim + item_dim + 6, n_hidden=n_hidden, hidden_size=hidden_size,
        dropout=dropout, l2_reg=l2_reg)

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


