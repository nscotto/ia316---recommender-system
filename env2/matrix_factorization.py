from surprise import SVD, Dataset, Reader


class MatrixFactorization(object):
    def __init__(self, clf=SVD()):
        self.clf = clf

    def train(self, X   ):
        reader = Reader(rating_scale=(1, 5))
        train_spr = Dataset.load_from_df(X, reader).build_full_trainset()
        self.clf.fit(train_spr)

    def predict(self, x):
        return self.clf.predict(uid=x['next_user'], iid=x['next_item'])

