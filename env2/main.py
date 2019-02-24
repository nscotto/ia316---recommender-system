import argparse
import requests
from feature_preparation import FeatureGenerator
from matrix_factorization import MatrixFactorization
from deep_factorization import DeepFactorization
from time import sleep
import numpy as np
# adress is http://35.180.254.42/reset?user_id=MC15CHD1JU24INEKKPU4



def main():
    # parser = argparse.ArgumentParser(description='main for second env')
    # parser.add_argument('--model', dest='model', type=str,
    #                     help='Model to use (deep or FM)')
    # args = parser.parse_args()
    # model_name = args.model
    user_id = 'MC15CHD1JU24INEKKPU4'
    adress = 'http://35.180.254.42'

    feature_gen = FeatureGenerator(user_id)
    feature_gen.reset()

    model = DeepFactorization(embedding_size=64,
                              n_user=feature_gen.nb_users,
                              n_item=feature_gen.nb_items,
                              batch_size=64)
    input = [feature_gen.X_train['user_id'],
             feature_gen.X_train['item_id'],
             feature_gen.X_train[['var0', 'var1', 'var2', 'var3', 'var4']]]
    model.train(input, feature_gen.y_train)

    next_u, next_i, next_var = feature_gen.next_user, feature_gen.next_item, \
        feature_gen.next_variables

    input_pred = [np.array([x]) for x in [next_i, next_i, next_var]]
    pred = model.predict(input_pred)[0][0]
    mse, mae = 0, 0
    i = 0

    while True:
        sleep(0.05)
        i += 1
        try:
            req = requests.get(adress + '/predict',
                               {'user_id': user_id, 'predicted_score': pred})
        except requests.exceptions.RequestException as e:
            print('The whole base has been parsed')
            break

        data = req.json()

        true_rating = data['rating']
        next_u, next_i, next_var = data['next_user'], data['next_item'], \
            data['next_variables']

        mse += (pred - true_rating) ** 2
        mae += abs(pred - true_rating)
        print('Iteration {}, SE={:.3f}'.format(i, (pred - true_rating)**2))

        input_pred = [np.array([x]) for x in [next_i, next_i, next_var]]
        pred = model.predict(input_pred)[0][0]

    print('Done \n MSE={:.3f} \n MAE={:.3f}'.format(mse/i, mae/i))

    return mse/i, mae/i


if __name__ == '__main__':
    main()