from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

class DeepFactorization(object):
    def __init__(self, embedding_size=64, n_user=64, n_item=32, vars=5,
                 batch_size=64, epochs=20, lr=0.01):
        self.batch_size = 64
        self.epochs = epochs
        self.lr = lr
        # For each sample we input the integer identifiers
        # of a single user and a single item
        user_id_input = Input(shape=[1], name='user')
        item_id_input = Input(shape=[1], name='item')
        variables_input = Input(shape=[vars], name='vars')

        embedding_size = 30
        user_embedding = Embedding(output_dim=embedding_size,
                                   input_dim=n_user + 1,
                                   input_length=1, name='user_embedding')(
            user_id_input)

        item_embedding = Embedding(output_dim=embedding_size,
                                   input_dim=n_item + 1,
                                   input_length=1, name='item_embedding')(
            item_id_input)

        # reshape from shape: (batch_size, input_length, embedding_size)
        # to shape: (batch_size, input_length * embedding_size) which is
        # equal to shape: (batch_size, embedding_size)
        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)

        input_vector = Concatenate()([user_vecs, item_vecs, variables_input])
        x = Dense(64, activation='relu')(input_vector)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
        y = Dense(1)(x)
        # y = Dot(axes=1)([user_vecs, item_vecs])


        self.model = Model(inputs=[user_id_input, item_id_input, variables_input],
                           outputs=y)
        self.model.compile(optimizer='adam', loss="mae")

    def train(self, X, y):
        self.hist_ = self.model.fit(X, y,
                                    batch_size=self.batch_size,
                                    epochs=self.epochs)
        return self

    def predict(self, x):
        return self.model.predict(x) #.reshape(-1, 1))

    def plot_train_history(self):
        plt.figure(figsize=(12, 12))
        plt.plot(self.hist_.history['loss'], label='train')
        plt.plot(self.hist_.history['val_loss'], label='val')
        plt.xlabel("number of epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()



