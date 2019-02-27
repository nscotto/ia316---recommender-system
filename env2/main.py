import argparse
import requests
from feature_preparation import FeatureGenerator
from matrix_factorization import MatrixFactorization
from deep_factorization import DeepFactorization
from time import sleep
import numpy as np
# adress is http://35.180.254.42/reset?user_id=MC15CHD1JU24INEKKPU4



def main():
    parser = argparse.ArgumentParser(description='main for second env')
    parser.add_argument('--embed-size', dest='n_embed', type=int,
                        help='size of the embeddings layers')
    parser.add_argument('--hidden-size', dest='n_hidden', type=int,
                        help='size of the hidden layers')
    args = parser.parse_args()
    embedding_size = args.n_embed
    hidden_size  = args.n_hidden
    user_id = 'MC15CHD1JU24INEKKPU4'
    adress = 'http://35.180.254.42'

    feature_gen = FeatureGenerator(user_id)
    feature_gen.reset()
    print('Number of users: {}\nNumber of items {}'.format(feature_gen.nb_users,
                                                           feature_gen.nb_items))

    model = DeepFactorization(embedding_size=embedding_size,
                              n_hidden=hidden_size,
                              n_user=feature_gen.nb_users,
                              n_item=feature_gen.nb_items,
                              batch_size=64)
    input = [feature_gen.X_train['user_id'],
             feature_gen.X_train['item_id'],
             feature_gen.X_train[['var0', 'var1', 'var2', 'var3', 'var4']]]
    model.train(input, feature_gen.y_train)

    next_u, next_i, next_var = feature_gen.next_user, feature_gen.next_item, \
        feature_gen.next_variables

    input_pred = [np.array([x]) for x in [next_u, next_i, next_var]]
    pred = model.predict(input_pred)[0][0]
    mse, mae = 0, 0
    i = 0

    n_iter=1000

    while True:
        sleep(0.05)
        i += 1
        try:
            req = requests.get(adress + '/predict',
                               {'user_id': user_id, 'predicted_score': pred})
        except requests.exceptions.RequestException as e:
            print('The whole base has been parsed')
            break
        # if i >= n_iter:
        #     break

        data = req.json()

        true_rating = data['rating']
        input_retrain = [np.array([x]) for x in [next_u, next_i, next_var]]
        model.train(input_retrain, np.array([true_rating]), 1, 0)

        next_u, next_i, next_var = data['next_user'], data['next_item'], \
            data['next_variables']

        mse += (pred - true_rating) ** 2
        mae += abs(pred - true_rating)
        if (i % 10) == 0:
            print('Iteration {}, SE={:.3f}'.format(i, (pred - true_rating)**2))

        input_pred = [np.array([x]) for x in [next_u, next_i, next_var]]
        pred = model.predict(input_pred)[0][0]

    print('Done \n MSE={:.3f} \n MAE={:.3f}'.format(mse/i, mae/i))

    return mse/i, mae/i


if __name__ == '__main__':
    main()