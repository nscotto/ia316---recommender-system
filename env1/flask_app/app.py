from flask import Flask, request, jsonify
from time import sleep
import requests

from utils import get_input, predict_value
from collab_filtering import df_to_ratings, predict_ratings_bias_sub

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


@app.route('/call', methods=['GET', 'POST'])
def call():
    """the function that makes the predictions"""
    adress = 'http://35.180.254.42'
    user_id = request.args.get('user_id')

    df, dic = get_input(user_id)
    next_u, next_i = dic['next_user'], dic['next_item']

    ratings = df_to_ratings(df)

    #collab filtering with bias
    pred_model = predict_ratings_bias_sub(ratings)
    pred = predict_value(next_u, next_i, pred_model)

    mse,mae = 0, 0
    i = 0.

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
        next_u, next_i = data['next_user'], data['next_item']

        mse += (pred - true_rating) ** 2
        mae += abs(pred - true_rating)
        print('iteration {} SE={:.3f}'.format(i, (pred-true_rating)**2))
        pred = predict_value(next_u, next_i, pred_model)

    print('Done \n MSE={:.3f} \n MAE={:.3f}'.format(mse/i, mae/i))

    return mse/i, mae/i




@app.route("/add", methods=['GET', 'POST'])
def predict():
    """ function to test flask server"""
    input1 = request.args.get('input1')
    input2 = request.args.get('input2')
    append = input1 + input2
    sum = float(input1) + float(input2)
    d = {'sum': sum, 'append': append}
    return jsonify(d)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
