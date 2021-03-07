---
title:  "Python으로 추천시스템 구현해보기"
search: true
categories:
  - Python
  - Recommend System
last_modified_at: 2021-03-05T
---

추천시스템은 데이터 사이언티스트에게는 듣기만 해도 굉장히 관심이가는 분야이다.
이 포스팅에서는 파이썬으로 구현한 협업필터링, 행렬요인화, 딥러닝 추천시스템 코드를 리뷰하며, 현업에서 직접 적용해볼 수 있는 내용을 공유하고자 한다.


#### Import Package


```python
import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Activation, Flatten, Concatenate, Embedding, Dot, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import RootMeanSquaredError
```


#### Load Dataset


```python
data = Dataset.load_builtin('ml-100k')

ratings = pd.DataFrame(data.raw_ratings, columns = ['user', 'item', 'rate',' id'])

print(ratings.shape)
ratings.head()
```


#### Score Mixin


```python
class ScoreMixin(object):

    def score(self):
        print(f"{self.__class__.__name__} evaluation start.")
        rating_test = self.ratings.sample(n=round(self.ratings.shape[0]*0.25))
        y_true = rating_test.rate
        y_pred = [self.rate(user, item) for user, item in zip(rating_test.user, rating_test.item)]
        rmse_score = self.rmse(y_true, y_pred)
        print(f"{self.__class__.__name__}", rmse_score)

        return rmse_score

    @staticmethod
    def rmse(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        return rmse

```


#### Baseline Recommend


```python

class Baseline_Recommender(ScoreMixin):

    def train(self, X):
        print(f"{self.__class__.__name__} train start.")
        self.ratings = X.copy()
        self.full_matrix = self.ratings.pivot('user', 'item', 'rate')
        self.full_matrix_dummy = self.full_matrix.fillna(0).copy()
        self.avg_rating = self.ratings.rate.mean()

        return self

    def rate(self, user, item):
        return self.avg_rating

    def recommend(self, user, n_items):
        print(f"{self.__class__.__name__} recommend start.")
        try:
            user_items = self.full_matrix.loc[user].copy()
        except: # recommend to random item for new customer
            reco_items = self.ratings.item.sample(n=n_items, random_state=42).tolist()
        else:
            for item in self.full_matrix.columns:
                if pd.notnull(user_items[item]):
                    user_items.loc[item] = 0 # already rating
                else:
                    user_items.loc[item] = self.rate(user, item)

            reco_items = user_items.sort_values(ascending=False).index[:n_items].tolist()

        return reco_items
```


#### UBCF (User Based Collaborative Filtering)

```python

class Ubcf_Recommender(Baseline_Recommender):

    def train(self, X):
        super().train(X)
        self.user_sim = pd.DataFrame(data=cosine_similarity(self.full_matrix_dummy, self.full_matrix_dummy),
                                     index=self.full_matrix_dummy.index,
                                     columns=self.full_matrix_dummy.index)

        return self

    def rate(self, user, item):
        if item in self.full_matrix.columns:
            user_sims = self.user_sim[user]
            item_ratings = self.full_matrix[item]
            none_ratings_idx = item_ratings[item_ratings.isnull()].index
            user_sims = user_sims.drop(none_ratings_idx)
            item_ratings = item_ratings.dropna()    
            rating = np.dot(user_sims, item_ratings) / user_sims.sum()
        else:
            rating = self.avg_rating

        return rating


```



#### UBCF KNN (K-Nearest Neighbor)

```python

class Ubcf_Knn_Recommender(Ubcf_Recommender):

    def __init__(self, K=30):
        self.K = K

    def rate(self, user, item):
        if item in self.full_matrix.columns:
            user_sims = self.user_sim[user]
            item_ratings = self.full_matrix[item]
            none_ratings_idx = item_ratings[item_ratings.isnull()].index
            user_sims = user_sims.drop(none_ratings_idx)
            item_ratings = item_ratings.dropna()    

            if self.K == 0:
                rating = np.dot(user_sims, item_ratings) / user_sims.sum()
            else:
                if len(user_sims) > 1:
                    neighbor_size = min(len(user_sims), self.K)
                    user_sims = np.array(user_sims)
                    item_ratings = np.array(item_ratings)
                    user_idx = np.argsort(user_sims)
                    user_sims = user_sims[user_idx][-neighbor_size:]
                    item_ratings = item_ratings[user_idx][-neighbor_size:]
                    rating = np.dot(user_sims, item_ratings) / user_sims.sum()
                else:
                    rating = self.avg_rating

        else:
            rating = self.avg_rating

        return rating

```

#### UBCF KNN Bias

```python

class Ubcf_Knn_Bias_Recommender(Ubcf_Knn_Recommender):

    def train(self, X):
        super().train(X)
        self.ratings_mean = self.full_matrix.mean(axis=1)
        self.ratings_bias = (self.full_matrix.T - self.ratings_mean).T

        return self

    def rate(self, user, item):
        if item in self.ratings_bias.columns:
            user_sims = self.user_sim[user]
            item_ratings = self.ratings_bias[item]
            none_ratings_idx = item_ratings[item_ratings.isnull()].index
            user_sims = user_sims.drop(none_ratings_idx)
            item_ratings = item_ratings.dropna()    

            if self.K == 0:
                rating = np.dot(user_sims, item_ratings) / user_sims.sum() + self.ratings_mean[user]
            else:
                if len(user_sims) > 1:
                    neighbor_size = min(len(user_sims), self.K)
                    user_sims = np.array(user_sims)
                    item_ratings = np.array(item_ratings)
                    user_idx = np.argsort(user_sims)
                    user_sims = user_sims[user_idx][-neighbor_size:]
                    item_ratings = item_ratings[user_idx][-neighbor_size:]
                    rating = np.dot(user_sims, item_ratings) / user_sims.sum() + self.ratings_mean[user]
                else:
                    rating = self.ratings_mean[user]

        else:
            rating = self.ratings_mean[user]

        return rating


```


#### UBCF KNN Bias Confidence level

```python

class Ubcf_Knn_Bias_Conf_Recommender(Ubcf_Knn_Bias_Recommender):

    def __init__(self, K=30, rating_cnt=2, min_rating=2):
        super().__init__(K)
        self.RATING_CNT = rating_cnt
        self.MIN_RATING = min_rating

    def train(self, X):
        super().train(X)
        rating_binary1 = np.array(self.full_matrix > 0).astype(float)
        rating_binary2 = rating_binary1.T
        counts = np.dot(rating_binary1, rating_binary2)
        self.counts = pd.DataFrame(counts, index=self.full_matrix.index, columns=self.full_matrix.index).fillna(0)

    def rate(self, user, item):

        if item in self.ratings_bias.columns:
            user_sims = self.user_sim[user]
            item_ratings = self.ratings_bias[item]

            no_ratings = item_ratings.isnull()
            common_counts = self.counts[user]
            low_significance = common_counts < self.RATING_CNT
            none_rating_idx = item_ratings[no_ratings | low_significance].index

            user_sims = user_sims.drop(none_rating_idx)
            item_ratings = item_ratings.dropna()

            if self.K == 0:
                rating = np.dot(user_sims, item_ratings) / user_sims.sum() + self.ratings_mean[user]
            else:
                if len(user_sims) > self.MIN_RATING:
                    neighbor_size = min(len(user_sims), self.K)
                    user_sims = np.array(user_sims)
                    item_ratings = np.array(item_ratings)
                    user_idx = np.argsort(user_sims)
                    user_sims = user_sims[user_idx][-neighbor_size:]
                    item_ratings =item_ratings[user_idx][-neighbor_size:]
                    rating = np.dot(user_sims, item_ratings) / user_sims.sum() + self.ratings_mean[user]
                else:
                    rating = self.ratings_mean[user]
        else:
            rating = self.ratings_mean[user]

        return rating

```



#### IBCF (Item Based Collaborative Filtering)

```python

class Ibcf_Recommender(Baseline_Recommender):

    def train(self, X):
        super().train(X)
        self.full_matrix_dummy = self.full_matrix_dummy.T
        self.item_sim = pd.DataFrame(data=cosine_similarity(self.full_matrix_dummy, self.full_matrix_dummy),
                                     index=self.full_matrix_dummy.index,
                                     columns=self.full_matrix_dummy.index)

        return self

    def rate(self, user, item):
        if item in self.full_matrix.columns:
            item_sims = self.item_sim[item]
            item_ratings = self.full_matrix.T[user]
            none_ratings_idx = item_ratings[item_ratings.isnull()].index
            item_sims = item_sims.drop(none_ratings_idx)
            item_ratings = item_ratings.dropna()
            rating = np.dot(item_sims, item_ratings) / item_sims.sum()
        else:
            rating = self.avg_rating

        return rating


```



#### MF (Matrix Factorization)

```python


class MF_Rrecommender(Baseline_Recommender):

    def __init__(self, K, alpha, beta, iter, verbose):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iter = iter
        self.verbose = verbose


    def train(self, X):
        super().train(X)
        self.num_users, self.num_items = self.full_matrix.shape
        self.P = np.random.normal(1/self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(1/self.K, size=(self.num_items, self.K))
        self.b_c = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = self.avg_rating

        self.R = np.array(self.full_matrix_dummy)
        self.user_to_idx = {user:idx for idx, user in enumerate(self.full_matrix.index)}
        self.item_to_idx = {item:idx for idx, item in enumerate(self.full_matrix.columns)}
        rows, cols = self.R.nonzero()

        self.samples = [(row, col, self.R[row, col]) for row, col, in zip(rows, cols)]

        for i in range(self.iter):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse_inner(self.samples)


            if self.verbose == True:
                if (i+1) % 10 == 0:
                    print(f"Iteration {i+1} : RMSE : {rmse}")

        self.R = ((self.P.dot(self.Q.T)).T + self.b_c).T + self.b_i + self.b
        self.full_matrix = pd.DataFrame(self.R, index=self.full_matrix.index, columns=self.full_matrix.columns)

        return self

    def sgd(self):
        for i, j, r in self.samples:
            pred = self.get_prediction(i, j)
            e = (r - pred)

            self.b_c[i] += self.alpha * (e - self.beta * self.b_c[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    def rmse_inner(self, samples):
        errors = []
        for x, y, r in samples:
            pred = self.get_prediction(x, y)
            error = r - pred
            errors.append(error)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))

        return rmse

    def get_prediction(self, i, j):
        pred = self.b + self.b_c[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)

        return pred

    def rate(self, user, item):
        i = self.user_to_idx[user]
        j = self.item_to_idx[item]

        return self.R[i,j]

```


#### Deep Learning (Neural Network)

```python

class OneLayerNeuralNetwork_Recommender(Baseline_Recommender):

    def __init__(self, K=200, epochs=100, batch_size=256):
        self.K = K
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X, refresh=True):
        super().train(X)
        self.user_to_idx = defaultdict(int)
        self.item_to_idx = defaultdict(int)

        def mapper(x, dict_val):
            if dict_val[x]:
                dict_val[x] = len(dict_val)

        self.ratings.user.apply(mapper, args=(self.user_to_idx,))
        self.ratings.item.apply(mapper, args=(self.item_to_idx,))

        self.ratings.user = self.ratings.user.apply(lambda x: self.user_to_idx[x])
        self.ratings.item = self.ratings.item.apply(lambda x: self.item_to_idx[x])
        M = self.ratings.user.unique().size + 1
        N = self.ratings.item.unique().size + 1
        self.mu = self.ratings.rate.mean()

        if refresh:
            user = Input(shape=(1,))
            item = Input(shape=(1,))

            P_embedding = Embedding(M, self.K, embeddings_regularizer=l2())(user)
            Q_embedding = Embedding(N, self.K, embeddings_regularizer=l2())(item)
            b_c_embedding = Embedding(M, 1, embeddings_regularizer=l2())(user)
            b_i_embedding = Embedding(N, 1, embeddings_regularizer=l2())(item)

            R = Dot(axes=2)([P_embedding, Q_embedding])
            R = Add()([R, b_c_embedding, b_i_embedding])
            R = Flatten()(R)

            self.model = Model(inputs=[user, item], outputs=R)
            self.model.compile(loss=self.rmse,
                               optimizer=SGD(),
                               metrics=[self.rmse, RootMeanSquaredError()])
            self.model.summary()
            self.model.fit(x=[self.ratings.user.values, self.ratings.item.values],
                           y=self.ratings.rate - self.mu,
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           validation_split=0.25)

            self.R = R

        return self

    def rate(self, user, item):
        u_idx = self.user_to_idx[user]
        i_idx = self.item_to_idx[item]

        return self.model.predict([np.array([u_idx]), np.array([i_idx])])[0][0] + self.mu

    @staticmethod
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    def save(self, name):
        self.model.save(name)



```




```python


class DeepNeuralNetwork_Recommender(Baseline_Recommender):

    def __init__(self, K=200, epochs=100, batch_size=256):
        self.K = K
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X):
        super().train(X)
        self.user_to_idx = defaultdict(int)
        self.item_to_idx = defaultdict(int)

        def mapper(x, dict_val):
            if not dict_val[x]:
                dict_val[x] = len(dict_val)

        self.ratings.user.apply(mapper, args=(self.user_to_idx,))
        self.ratings.item.apply(mapper, args=(self.item_to_idx,))

        self.ratings.user = self.ratings.user.apply(lambda x: self.user_to_idx[x])
        self.ratings.item = self.ratings.item.apply(lambda x: self.item_to_idx[x])
        M = self.ratings.user.unique().size + 1
        N = self.ratings.item.unique().size + 1
        self.mu = self.ratings.rate.mean()

        user = Input(shape=(1,), name='UserInput')
        item = Input(shape=(1,), name='ItemInput')

        P_embedding = Embedding(M, self.K, embeddings_regularizer=l2(), name='P')(user)
        Q_embedding = Embedding(N, self.K, embeddings_regularizer=l2(), name='Q')(item)
        user_bias = Embedding(M, 1, embeddings_regularizer=l2(), name='UserBias')(user)
        item_bias = Embedding(N, 1, embeddings_regularizer=l2(), name='ItemBias')(item)

        P_flat = Flatten(name='P_flat')(P_embedding)
        Q_flat = Flatten(name='Q_falt')(Q_embedding)
        user_bias_flat = Flatten(name='UserBias_flat')(user_bias)
        item_bias_flat = Flatten(name='ItemBias_flat')(item_bias)

        R = Concatenate(name='R_concat')([P_flat, Q_flat, user_bias_flat, item_bias_flat])

        for emb in [2048, 512, 128, 1]:
            R = Dense(emb)(R)
            R = Activation('linear')(R) if emb != 1 else R

        self.model = Model(inputs=[user, item], outputs=R)
        self.model.compile(loss=self.rmse,
                           optimizer=SGD(),
                           metrics=[self.rmse, RootMeanSquaredError()])
        self.model.summary()

        self.model.fit(
            x=[self.ratings.user.values, self.ratings.item.values],
            y=self.ratings.rate - self.mu,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.25
        )

        self.R = R

        return self

    @staticmethod
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    def save(self, name):
        self.model.save(name)



```




#### Compare RMSE


```python


models = {
    "baseline": Baseline_Recommender(),
    "ubcf": Ubcf_Recommender(),
    "ubcf_knn": Ubcf_Knn_Recommender(K=30),
    "ubcf_knn_bias": Ubcf_Knn_Bias_Recommender(K=30),
    "ubcf_knn_bias_conf": Ubcf_Knn_Bias_Conf_Recommender(K=30, rating_cnt=2, min_rating=2),
    "ibdf": Ibcf_Recommender(),
    "mf": MF_Rrecommender(K=30, alpha=0.001, beta=0.02, iter=100, verbose=True),
    "olnnr": OneLayerNeuralNetwork_Recommender(),
    "dnnr": DeepNeuralNetwork_Recommender()
}

scores = dict()
for name, model in models.items():
    model.train(ratings)
    scores[name] = model.score()



```
