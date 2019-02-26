from recommender_evaluation import *

from deep_implicit_feedback_recsys import *

if __name__ == '__main__':
    ID = 'VI2X71V0287S9F9B7SCU'
    mean, rate = eval_recommender(ImplicitRecommenderWithNull_no_indicator,
            n_pred=10, online_batch_size=None)
    print(mean)
    print(rate)
