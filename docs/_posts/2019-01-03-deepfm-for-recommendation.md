---
title: "Implement DeepFM model in Keras"
date: 2019-01-03 11:39:00 +0800
categories: 
  - deep-learning
  - recsys
tags:
  - keras
  - python
toc: true
toc_sticky: true
---

# Introduction
Wide and deep architect has been proven as one of deep learning applications combining memorization and generatlization in areas such as search and recommendation. Google released its [wide&deep learning](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) in 2016. 

* wide part: helps to memorize the past behaviour for specific choice
* deep part: embed into low dimension, help to discover new user, product combinations

Later, on top of wide & deep learning, [deepfm](https://arxiv.org/abs/1703.04247) was developed combining DNN model and Factorization machines, to furthur address the interactions among the features. 

## wide & deep model
![wide&deep learning](https://1.bp.blogspot.com/-Dw1mB9am1l8/V3MgtOzp3uI/AAAAAAAABGs/mP-3nZQCjWwdk6qCa5WraSpK8A7rSPj3ACLcB/s640/image04.png)

## deepFM model
![deepfm learning](https://www.researchgate.net/profile/Huifeng_Guo/publication/318829508/figure/fig1/AS:522607722467328@1501610798143/Wide-deep-architecture-of-DeepFM-The-wide-and-deep-component-share-the-same-input-raw.png)

## Comparison
wide&deep learning is logistic regression + deep neural network. In wide part of wide & deep learning, it is a logistic regression, which requires a lot of manual feature engineering efforts to generate the large-scale feature set for wide part.    
While the deepfm model instead is factorization machines + deep neural network, as known as neural factorization machines. 
DeepFM has

* 1st order embeded layers to have overall characterization of indiviual features.
\begin{equation*}
y = \sum{w_ix_i}
\end{equation*}
* 2nd order shared embeded layers for both deep and fm parts, from which dot product between pairs of embeded features address the 2nd order feature interactions.
\begin{equation*}
y = \sum{w_{i,j}x_ix_j}
\end{equation*}
If the only the factorization machines part is kept, it will reduce to neural collabrative filtering. 

## DeepFM model in details
### 1st order factorization machines (summation of all 1st order embed layers)
+ numeric features with shape (None, 1) => dense layer => map to shape (None, 1)
+ categorical features (single level) with shape (None,1) => embedding layer (latent_dim = 1) => map to shape (None, 1)
+ categorical features (multi level) with shape (None,L) => embedding layer (latent_dim = 1) => map to shape (None, L)
+ output will summation of all embeded features, result in a tensor with shape (None, 1)
### 2nd order factorization machines (summation of dot product between 2nd order embed layers)
+ numeric features => dense layer => map to shape (None, 1, k)
+ categorical features (single level) => embedding layer (latent_dim = k) => map to shape (None, 1, k)
+ categorical features (multi level) with shape (None,L) => embedding layer (latent_dim = k) => map to shape (None, L, k)
+ shared embed layer will be the concatenated layers of all embeded features
+ shared embed layer => dot layer => 2nd order of fm part
### deep part (DNN model on shared embed layers)
+ shared embed layer => series of dense layers => deep part
    
## preprocess data

The dataset used to implement deepfm is movieLens(ml-1m) data.    
To add more features to the `ratings.dat`, I joined the user features and movies features.
The features used are as below:
* numeric feature: user_fea3
* categorical feature (single level): uid, mid
* categorical feature (multi level): movie_genre

movie genre is a mutli-value field delimited by `'|'`.   
Multi-Hot encoding of this field is done by kera text `Tokenizer`. 

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_ratings():
    COL_NAME = ['uid','mid','rating','timestamp']
    df = pd.read_csv('./dataset/ml-1m/ratings.dat',sep='::', header=None, engine='python', names=COL_NAME)
    return df

def load_movies():
    COL_NAME = ['mid','movie_name','movie_genre']
    df = pd.read_csv('./dataset/ml-1m/movies.dat',sep='::', header=None, engine='python', names=COL_NAME)
    return df

def load_users():
    COL_NAME = ['uid','user_fea1','user_fea2','user_fea3','user_fea4']
    df = pd.read_csv('./dataset/ml-1m/users.dat',sep='::', header=None, engine='python', names=COL_NAME)
    return df

def text2seq(text, n_genre):
    """ using tokenizer to encoded the multi-level categorical feature
    """
    tokenizer = Tokenizer(lower=True, split='|',filters='', num_words=n_genre)
    tokenizer.fit_on_texts(text)
    seq = tokenizer.texts_to_sequences(text)
    seq = pad_sequences(seq, maxlen=3,padding='post')
    return seq

n_genre = 15

ratings = load_ratings()
movies = load_movies()
users = load_users()

print("====== rating.dat ======")
print(ratings.head())
print("===== movies.dat ======")
print(movies.head())
print("====== users.dat ======")
print(users.head())

movies['movie_genre'] = text2seq(movies.movie_genre.values, n_genre=n_genre).tolist()

ratings = ratings.join(movies.set_index('mid'), on = 'mid', how = 'left')
ratings = ratings.join(users.set_index('uid'), on = 'uid', how = 'left')
print("====== preprocessed data =======")
(ratings.head())
```

```
====== rating.dat ======
   uid   mid  rating  timestamp
0    1  1193       5  978300760
1    1   661       3  978302109
2    1   914       3  978301968
3    1  3408       4  978300275
4    1  2355       5  978824291
===== movies.dat ======
   mid                          movie_name                   movie_genre
0    1                    Toy Story (1995)   Animation|Children's|Comedy
1    2                      Jumanji (1995)  Adventure|Children's|Fantasy
2    3             Grumpier Old Men (1995)                Comedy|Romance
3    4            Waiting to Exhale (1995)                  Comedy|Drama
4    5  Father of the Bride Part II (1995)                        Comedy
====== users.dat ======
   uid user_fea1  user_fea2  user_fea3 user_fea4
0    1         F          1         10     48067
1    2         M         56         16     70072
2    3         M         25         15     55117
3    4         M         45          7     02460
4    5         M         25         20     55455
====== preprocessed data =======
	uid	mid	rating	timestamp	movie_name	movie_genre	user_fea1	user_fea2	user_fea3	user_fea4
0	1	1193	5	978300760	One Flew Over the Cuckoo's Nest (1975)	[1, 0, 0]	F	1	10	48067
1	1	661	3	978302109	James and the Giant Peach (1996)	[9, 13, 0]	F	1	10	48067
2	1	914	3	978301968	My Fair Lady (1964)	[13, 5, 0]	F	1	10	48067
3	1	3408	4	978300275	Erin Brockovich (2000)	[1, 0, 0]	F	1	10	48067
4	1	2355	5	978824291	Bug's Life, A (1998)	[9, 2, 0]	F	1	10	48067
```

## Construct model

There are 3 parts of the deepFM models:

* 1st order factorization machines
* 2nd order factorization machines
* deep neural network

Let's start with input layer definition.

### define input layers

For dataset mixed with numeric and categerical features, they need to be treated differently.
* numeric features can be concatenated to inputs, with shape (None, num_of_numeric)
* categorical features can be encoded individually to inputs, with shape (None, 1) each.

```python
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def define_input_layers():
    # numerica features
    fea3_input = Input((1,), name = 'input_fea3')
    num_inputs = [fea3_input]
    # single level categorical features
    uid_input = Input((1,), name = 'input_uid')
    mid_input = Input((1,), name= 'input_mid')
    cat_sl_inputs = [uid_input, mid_input]

    # multi level categorical features (with 3 genres at most)
    genre_input = Input((3,), name = 'input_genre')
    cat_ml_inputs = [genre_input]

    inputs = num_inputs + cat_sl_inputs + cat_ml_inputs
    
    return inputs

inputs = define_input_layers()
```

### 1st order factorization machines

1st order will require features to map to a scalar.
so for
* numeric feature: a dense layer will convert tensor to shape (None,1)
* categorical feature: a embedding layer will convert tensor to shape (None,1,1) and then reshape layer to reshape to (None,1)

```python
def Tensor_Mean_Pooling(name = 'mean_pooling', keepdims = False):
    return Lambda(lambda x: K.mean(x, axis = 1, keepdims=keepdims), name = name)

def fm_1d(inputs, n_uid, n_mid, n_genre):
    
    fea3_input, uid_input, mid_input, genre_input = inputs
    
    # all tensors are reshape to (None, 1)
    num_dense_1d = [Dense(1, name = 'num_dense_1d_fea4')(fea3_input)]
    cat_sl_embed_1d = [Embedding(n_uid + 1, 1, name = 'cat_embed_1d_uid')(uid_input),
                        Embedding(n_mid + 1, 1, name = 'cat_embed_1d_mid')(mid_input)]
    cat_ml_embed_1d = [Embedding(n_genre + 1, 1, mask_zero=True, name = 'cat_embed_1d_genre')(genre_input)]

    cat_sl_embed_1d = [Reshape((1,))(i) for i in cat_sl_embed_1d]
    cat_ml_embed_1d = [Tensor_Mean_Pooling(name = 'embed_1d_mean')(i) for i in cat_ml_embed_1d]
    
    # add all tensors
    y_fm_1d = Add(name = 'fm_1d_output')(num_dense_1d + cat_sl_embed_1d + cat_ml_embed_1d)
    
    return y_fm_1d

y_1d = fm_1d(inputs, 10, 10, 10)
```
![](https://github.com/6chaoran/recsys/raw/4eeb08cf7d103cf699cdca926eea9d0ecd4f1a61/deepfm/image/fm_model_1d.png)

### 2nd order factorization machines
In 2nd order FM, each feature is map to shape (None, 1, k) and then stack to `concat_embed_2d` layer with shape (None, p, k).    
k - matrix factorization latent dimension, p is feature dimension.

the calculation of interaction terms can be simplified, using    
\begin{equation*}
\sum{x_ix_j} = \frac{1}{2} \left((\sum{x})^2 - \sum({x}^2)\right)
\end{equation*}

Hence, the sum of 2nd order interactions = square of sum of `concat_embed_2d` - sum of squared `concat_embed_2d` in p dimension, the resulting tensor will have a shape (None, k)

```python
def fm_2d(inputs, n_uid, n_mid, n_genre, k):
    
    fea3_input, uid_input, mid_input, genre_input = inputs
    
    num_dense_2d = [Dense(k, name = 'num_dense_2d_fea3')(fea3_input)] # shape (None, k)
    num_dense_2d = [Reshape((1,k))(i) for i in num_dense_2d] # shape (None, 1, k)

    cat_sl_embed_2d = [Embedding(n_uid + 1, k, name = 'cat_embed_2d_uid')(uid_input), 
                       Embedding(n_mid + 1, k, name = 'cat_embed_2d_mid')(mid_input)] # shape (None, 1, k)
    
    cat_ml_embed_2d = [Embedding(n_genre + 1, k, name = 'cat_embed_2d_genre')(genre_input)] # shape (None, 3, k)
    cat_ml_embed_2d = [Tensor_Mean_Pooling(name = 'cat_embed_2d_genure_mean', keepdims=True)(i) for i in cat_ml_embed_2d] # shape (None, 1, k)

    # concatenate all 2d embed layers => (None, ?, k)
    embed_2d = Concatenate(axis=1, name = 'concat_embed_2d')(num_dense_2d + cat_sl_embed_2d + cat_ml_embed_2d)

    # calcuate the interactions by simplication
    # sum of (x1*x2) = sum of (0.5*[(xi)^2 - (xi^2)])
    tensor_sum = Lambda(lambda x: K.sum(x, axis = 1), name = 'sum_of_tensors')
    tensor_square = Lambda(lambda x: K.square(x), name = 'square_of_tensors')

    sum_of_embed = tensor_sum(embed_2d)
    square_of_embed = tensor_square(embed_2d)

    square_of_sum = Multiply()([sum_of_embed, sum_of_embed])
    sum_of_square = tensor_sum(square_of_embed)

    sub = Subtract()([square_of_sum, sum_of_square])
    sub = Lambda(lambda x: x*0.5)(sub)
    y_fm_2d = Reshape((1,), name = 'fm_2d_output')(tensor_sum(sub))
    
    return y_fm_2d, embed_2d

y_fm2_d, embed_2d = fm_2d(inputs, 10, 10, 10, 5)
```

![](https://github.com/6chaoran/recsys/raw/4eeb08cf7d103cf699cdca926eea9d0ecd4f1a61/deepfm/image/fm_model_2d.png)

### deep part

this part is simply a DNN framework with input as `concat_embed_2d` layer

```python
def deep_part(embed_2d, dnn_dim, dnn_dr):
    
    # flat embed layers from 3D to 2D tensors
    y_dnn = Flatten(name = 'flat_embed_2d')(embed_2d)
    for h in dnn_dim:
        y_dnn = Dropout(dnn_dr)(y_dnn)
        y_dnn = Dense(h, activation='relu')(y_dnn)
    y_dnn = Dense(1, activation='relu', name = 'deep_output')(y_dnn)
    
    return y_dnn

y_dnn = deep_part(embed_2d, [16, 16], 0.5)
```
![](https://github.com/6chaoran/recsys/raw/4eeb08cf7d103cf699cdca926eea9d0ecd4f1a61/deepfm/image/deep_model.png)


### Put Together All Parts

```python
def deep_fm_model(n_uid, n_mid, n_genre, k, dnn_dim, dnn_dr):
    
    inputs = define_input_layers()
    
    y_fm_1d = fm_1d(inputs, n_uid, n_mid, n_genre)
    y_fm_2d, embed_2d = fm_2d(inputs, n_uid, n_mid, n_genre, k)
    y_dnn = deep_part(embed_2d, dnn_dim, dnn_dr)
    
    # combinded deep and fm parts
    y = Concatenate()([y_fm_1d, y_fm_2d, y_dnn])
    y = Dense(1, name = 'deepfm_output')(y)
    
    fm_model_1d = Model(inputs, y_fm_1d)
    fm_model_2d = Model(inputs, y_fm_2d)
    deep_model = Model(inputs, y_dnn)
    deep_fm_model = Model(inputs, y)
    
    return fm_model_1d, fm_model_2d, deep_model, deep_fm_model
```

![](https://github.com/6chaoran/recsys/raw/4eeb08cf7d103cf699cdca926eea9d0ecd4f1a61/deepfm/image/deep_fm_model.png)

generate models with configuration

```python
params = {
    'n_uid': ratings.uid.max(),
    'n_mid': ratings.mid.max(),
    'n_genre': 14,
    'k':20,
    'dnn_dim':[64,64],
    'dnn_dr': 0.5
}

fm_model_1d, fm_model_2d, deep_model, deep_fm_model = deep_fm_model(**params)
```

## Split Training & Validation Data

```python
def df2xy(ratings):
    x = [ratings.user_fea3.values, 
         ratings.uid.values, 
         ratings.mid.values, 
         np.concatenate(ratings.movie_genre.values).reshape(-1,3)]
    y = ratings.rating.values
    return x,y

in_train_flag = np.random.random(len(ratings)) <= 0.9
train_data = ratings.loc[in_train_flag,]
valid_data = ratings.loc[~in_train_flag,]
train_x, train_y = df2xy(train_data)
valid_x, valid_y = df2xy(valid_data)
```

## Train Model

```python
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
# train  model
deep_fm_model.compile(loss = 'MSE', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=3)
model_ckp = ModelCheckpoint(filepath='./model/deepfm_weights.h5', 
                            monitor='val_loss',
                            save_weights_only=True, 
                            save_best_only=True)
callbacks = [model_ckp,early_stop]
train_history = deep_fm_model.fit(train_x, train_y, 
                                  epochs=30, batch_size=2048, 
                                  validation_split=0.1, 
                                  callbacks = callbacks)
```

```
Train on 810025 samples, validate on 90003 samples
Epoch 1/30
810025/810025 [==============================] - 8s 10us/step - loss: 2.3674 - val_loss: 3.6052
Epoch 2/30
810025/810025 [==============================] - 6s 8us/step - loss: 0.9796 - val_loss: 2.8897
Epoch 3/30
810025/810025 [==============================] - 6s 8us/step - loss: 0.9079 - val_loss: 2.4225
Epoch 4/30
810025/810025 [==============================] - 6s 8us/step - loss: 0.8677 - val_loss: 2.1066
Epoch 5/30
810025/810025 [==============================] - 6s 8us/step - loss: 0.8464 - val_loss: 1.8863
```

```python
%matplotlib inline
pd.DataFrame(train_history.history).plot()
```
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXcAAAD8CAYAAACMwORRAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3Xd0HOW9//H3d4t21YtVLKsaF2xsuSGblhgDoQQIJSHBhNB+SbgBQhrk5vJLgySc3EsSuOGGC5eEmh/FXEILBAjFiTEhxrIt9wK4yrItWbaKy6rsPr8/ZiWvZclquxrt7Pd1zpydnR3Nfsd7/JnZZ559RowxKKWUchaX3QUopZSKPg13pZRyIA13pZRyIA13pZRyIA13pZRyIA13pZRyIA13pZRyIA13pZRyIA13pZRyII9db5ybm2vKy8vtenullIpLy5Yt22uMyetrPdvCvby8nKqqKrveXiml4pKIbOvPetoso5RSDqThrpRSDqThrpRSDmRbm7tSKjG1t7dTU1NDIBCwu5QRze/3U1xcjNfrHdTfa7grpYZVTU0N6enplJeXIyJ2lzMiGWNoaGigpqaGsWPHDmob2iyjlBpWgUCAUaNGabAfh4gwatSoIX270XBXSg07Dfa+DfXfKO7CfePuFu5+bR2H24J2l6KUUiNW3IV7zf5D/P69LaysabS7FKWUGrHiLtxPLssGYNm2/TZXopRKBGlpab2+tnXrVqZOnTqM1fRf3IV7VkoSEwvSWLp1n92lKKXUiBWXXSFPLsvh1VW1BEMGt0svzCgVr+7681rW1TZHdZsnjcngp5+b0uvrP/jBDygrK+Pmm28G4M4770REWLRoEfv376e9vZ1f/OIXXHrppQN630AgwE033URVVRUej4d7772Xs846i7Vr13LDDTfQ1tZGKBTiT3/6E2PGjOFLX/oSNTU1BINBfvzjH3PllVcOab+76/PMXUT8IvKhiKwUkbUiclcP61wvIvUiUh2evhbVKruZXZ5NS6CDTXtaYvk2SikHmj9/PgsWLOh6/txzz3HDDTfw4osvsnz5chYuXMhtt92GMWZA233ggQcAWL16Nc888wzXXXcdgUCAhx56iG9/+9tUV1dTVVVFcXExb7zxBmPGjGHlypWsWbOGCy64IKr7CP07c28FzjbGHBARL7BYRF43xvyz23oLjDHfjHqFPZhdngNA1bb9TC7MGI63VErFwPHOsGNl5syZ1NXVUVtbS319PdnZ2RQWFvLd736XRYsW4XK52LlzJ3v27GH06NH93u7ixYu59dZbAZg0aRJlZWVs2rSJ0047jbvvvpuamho+//nPM2HCBCoqKrj99tv5wQ9+wMUXX8ynP/3pqO9nn2fuxnIg/NQbngZ2SIuy4uxk8tN9VGm7u1JqEK644gqef/55FixYwPz583nqqaeor69n2bJlVFdXU1BQMOAfEPV2pv/lL3+ZV155heTkZM4//3zeffddJk6cyLJly6ioqOCOO+7gZz/7WTR26yj9uqAqIm4RqQbqgLeMMUt6WO0LIrJKRJ4XkZJetnOjiFSJSFV9ff2gixYRZpfnULVVe8wopQZu/vz5PPvsszz//PNcccUVNDU1kZ+fj9frZeHChWzb1q8h048yd+5cnnrqKQA2bdrE9u3bOfHEE9m8eTMnnHAC3/rWt7jkkktYtWoVtbW1pKSk8JWvfIXbb7+d5cuXR3sX+xfuxpigMWYGUAzMEZHufX/+DJQbY6YBbwNP9LKdh40xlcaYyry8Pm8kclyV5dnsbDxMbePhIW1HKZV4pkyZQktLC0VFRRQWFnL11VdTVVVFZWUlTz31FJMmTRrwNm+++WaCwSAVFRVceeWVPP744/h8PhYsWMDUqVOZMWMGGzZs4Nprr2X16tXMmTOHGTNmcPfdd/OjH/0o6vsoA71oICI/BQ4aY37dy+tuYJ8xJvN426msrDRDuRPTmp1NXPxfi7n/qplcMn3MoLejlBpe69evZ/LkyXaXERd6+rcSkWXGmMq+/rY/vWXyRCQrPJ8MfAbY0G2dwoinlwDr+1H3kEwanU5Kklvb3ZVSqgf96S1TCDwRPiN3Ac8ZY14VkZ8BVcaYV4BvicglQAewD7g+VgV38rhdzCrN1nZ3pVTMrV69mmuuueaoZT6fjyVLerr8ODL0Ge7GmFXAzB6W/yRi/g7gjuiW1rfK8mzuf+cjmgPtZPgHN6C9Ukr1paKigurqarvLGJC4G34gUmVZDiEDK7brIGJKKRUprsN9RmkWbpdou7tSSnUT1+Ge5vNwUmGGtrsrpVQ3cR3uYLW7r9ixn/ZgyO5SlFJx4njD+DpF/Id7WQ6B9hBrozyynFJKxbP4D/dy6+Yd2u6ulBooYwzf//73mTp1KhUVFV2jRe7atYu5c+cyY8YMpk6dynvvvUcwGOT666/vWve+++6zufrji8vx3CMVZPgpzUmhaut+vhb9gdWUUrH0+r/B7tXR3eboCvjsv/dr1RdeeIHq6mpWrlzJ3r17mT17NnPnzuXpp5/m/PPP54c//CHBYJBDhw5RXV3Nzp07WbNmDQCNjSO7l17cn7kDVJZlU7Vt34DHX1ZKJbbFixdz1VVX4Xa7KSgo4Mwzz2Tp0qXMnj2bxx57jDvvvJPVq1eTnp7OCSecwObNm7n11lt54403yMgY2cONx/2ZO0BleQ4vrNjJ1oZDjM1NtbscpVR/9fMMO1Z6OyGcO3cuixYt4rXXXuOaa67h+9//Ptdeey0rV67kzTff5IEHHuC5557j0UcfHeaK+88RZ+6ztd1dKTUIc+fOZcGCBQSDQerr61m0aBFz5sxh27Zt5Ofn8/Wvf52vfvWrLF++nL179xIKhfjCF77Az3/+85gM0xtNjjhzH5eXRlaKl6qt+/liZY9DySul1DEuv/xyPvjgA6ZPn46IcM899zB69GieeOIJfvWrX+H1eklLS+PJJ59k586d3HDDDYRCVrfrX/7ylzZXf3wDHvI3WoY65G93X318KVsaDvLubfOitk2lVPTpkL/9F9Mhf+NFZXkOm+sP0nCg1e5SlFLKdo4J985292XbdCgCpZRyTLhPLcokye2iSsNdqRFPuy33baj/Ro4Jd7/XzbTiTO0xo9QI5/f7aWho0IA/DmMMDQ0N+P3+QW/DEb1lOlWW5/DI4s0E2oP4vW67y1FK9aC4uJiamhrq6+vtLmVE8/v9FBcXD/rvHRXus8uzeejvhpU7GjnlhFF2l6OU6oHX62Xs2LF2l+F4jmmWATi5LPxjJm13V0olOEeFe1ZKEhPy07TdXSmV8BwV7mC1u1dt208opBdrlFKJy3nhXpZNS6CDTXUtdpeilFK2cVy4zy7PAdD7qiqlEprjwr0kJ5n8dJ+2uyulElqf4S4ifhH5UERWishaEbmrh3V8IrJARD4WkSUiUh6LYvtDRJhdnsNSPXNXSiWw/py5twJnG2OmAzOAC0Tk1G7rfBXYb4wZD9wH/Ed0yxyYk8uy2dl4mF1Nh+0sQymlbNNnuBvLgfBTb3jq3hXlUuCJ8PzzwDkiIlGrcoC03V0plej61eYuIm4RqQbqgLeMMUu6rVIE7AAwxnQATYBtPxGdXJhOSpJb292VUgmrX+FujAkaY2YAxcAcEZnabZWeztKP6WguIjeKSJWIVMVyXAmP28XM0ixtd1dKJawB9ZYxxjQCfwMu6PZSDVACICIeIBM45rTZGPOwMabSGFOZl5c3qIL7q7Ishw27m2kJtMf0fZRSaiTqT2+ZPBHJCs8nA58BNnRb7RXguvD8FcC7xubxPGeX5xAysGJ7o51lKKWULfpz5l4ILBSRVcBSrDb3V0XkZyJySXidR4BRIvIx8D3g32JTbv/NKM3C7RJtd1dKJaQ+h/w1xqwCZvaw/CcR8wHgi9EtbWjSfB4mF6brCJFKqYTkuF+oRqosy2HF9kbagyG7S1FKqWHl6HCfXZ7D4fYg62qb7S5FKaWGlaPDvbLcunnHPz5psLkSpZQaXo4O94IMPzNKsnhlZa3dpSil1LBydLgDXD6ziPW7mtm4W8d3V0olDseH+0XTCnG7hJeqd9pdilJKDRvHh3tumo+5E3J5ecVOvfWeUiphOD7cAS6bWURtU4Cl+oMmpVSCSIhwP/ekAlKS3No0o5RKGAkR7ilJHs6fMprXVu2itSNodzlKKRVzCRHuAJfOGENzoIO/bYzdUMNKKTVSJEy4f2p8LrlpSby0QptmlFLOlzDh7nG7uHjaGN7ZUEfTYR3jXSnlbAkT7mD9oKmtI8Qba3bZXYpSSsVUQoX7tOJMxuam8tIKHY5AKeVsCRXuIsJlM4r455YGdjUdtrscpZSKmYQKd7B6zRgDr1Tr2btSyrkSLtzLc1OZUZLFi9prRinlYAkX7mBdWN2wu0VHilRKOVZChruOFKmUcrqEDHcdKVIp5XQJGe6gI0UqpZwtYcNdR4pUSjlZwoa7jhSplHKyhA13ODJS5MINOlKkUspZ+gx3ESkRkYUisl5E1orIt3tYZ56INIlIdXj6SWzKja7OkSJf1qYZpZTDePqxTgdwmzFmuYikA8tE5C1jzLpu671njLk4+iXGTudIkU8v2U7T4XYyk712l6SUUlHR55m7MWaXMWZ5eL4FWA8Uxbqw4XL5zCLagjpSpFLKWQbU5i4i5cBMYEkPL58mIitF5HURmRKF2oaFjhSplHKifoe7iKQBfwK+Y4xp7vbycqDMGDMd+C/gpV62caOIVIlIVX39yLiIqSNFKqWcqF/hLiJerGB/yhjzQvfXjTHNxpgD4fm/AF4Rye1hvYeNMZXGmMq8vLwhlh49OlKkUspp+tNbRoBHgPXGmHt7WWd0eD1EZE54uw3RLDSWynNTmVmqI0UqpZyjP2fuZwDXAGdHdHW8UES+ISLfCK9zBbBGRFYC9wPzjTFxNWjLZTOskSI37O7e4qSUUvGnz66QxpjFgPSxzu+A30WrKDtcNK2Qn726jhdX7OSOz2bYXY5SSg1JQv9CNVJumo9zJuXzzJLtNB1qt7scpZQaEg33CN89dyItrR08tOgTu0tRSqkh0XCPMLkwg0umj+Gx97dQ1xywuxyllBq0+Av3DX+BX0+E/VtjsvnvnTuRjqDh/nc/isn2lVJqOMRfuCelwIE90LgjJpsvG5XK/DklPPvhDrY1HIzJeyilVKzFX7hnlliPTbEJd4BvnT0Bj1u4761NMXsPpZSKpfgL94zwmGUxOnMHyM/wc/3pY3l5ZS3rd2m/d6VU/Im/cPf6Ia0gpmfuADedOY40n4dfv7kxpu+jlFKxEH/hDlbTTIzDPTPFyzfOHMc7G+qo0ptoK6XiTHyGe1ZJTJtlOt1wRjm5aT7ueXMjcTaaglIqwcVnuGeWQFMNhEIxfZuUJA/fOmc8H27Zx983jYwhipVSqj/iN9yDrXAw9oE7f3YpJTnJ/OrNjYRCevaulIoP8RnuWZ3dIWti/lZJHhffO3cia2ubeW213opPKRUf4jPcu/q6bx+Wt7tkehEnFqRz71ubaA/GtilIKaWiIT7DvfPMfRguqgK4XcLt55/Ilr0HeX5Z7L8tKKXUUMVnuPszwZcR8+6QkT4zOZ9ZpVn89u2PCLQHh+19lVJqMOIz3OFIj5lhIiJ8//xJ7G4O8McPtg3b+yql1GDEb7gPU1/3SKeNG8XciXk88LePaQ7oDT2UUiNX/IZ7ZsmwXVCN9K/nn0jjoXb+sGjzsL+3Ukr1VxyHezEEmiAwvAN7TS3K5KKKQv6weAt7D7QO63srpVR/xW+4D2Nf9+6+d95EWjtCPLDw42F/b6WU6o/4DffMUutxGHvMdBqXl8YXTy7mjx9sY8X2/cP+/kop1Zf4Dfeuvu7D3+4OcMdnJzM60883n15B46E2W2pQSqnexG+4p+aDO8mWM3ewhgR+4MuzqGsJcNtzK3XcGaXUiBK/4e5yWXdlGubukJGml2Txwwsn886GOn7/nvaeUUqNHH2Gu4iUiMhCEVkvImtF5Ns9rCMicr+IfCwiq0RkVmzK7SZreH/I1JPrTi/nworR3PPmRpbqTT2UUiNEf87cO4DbjDGTgVOBW0TkpG7rfBaYEJ5uBB6MapW9ySy1rVmmk4jw71+YRkl2Mt98ejkN2j1SKTUC9Bnuxphdxpjl4fkWYD1Q1G21S4EnjeWfQJaIFEa92u6ySqBlN3TYe0Ezw+/lgatnsf9QO99ZUE1Q29+VUjYbUJu7iJQDM4El3V4qAiJPoWs49gAQfZnFgIFm+0dqnDImk7sumcJ7H+3V/u9KKdv1O9xFJA34E/AdY0z3n4VKD39yzOmriNwoIlUiUlVfH4W7KGXa90OmnsyfXcLlM4u47+1NvP/xXrvLUUolsH6Fu4h4sYL9KWPMCz2sUgOURDwvBmq7r2SMedgYU2mMqczLyxtMvUcb5nHd+yIi/OKyqYzLS+Pbz66grjlgd0lKqQTVn94yAjwCrDfG3NvLaq8A14Z7zZwKNBljYn9PuowiQGy/qBop1efhwatncbA1yK3PrKBD79yklLJBf87czwCuAc4WkerwdKGIfENEvhFe5y/AZuBj4PfAzbEptxuPD9IKRsyZe6cJBencfflUlmzZx31vb7K7HKVUAvL0tYIxZjE9t6lHrmOAW6JV1IBk2TP0b18+P6uYD7fs44GFn1BZnsNZJ+bbXZJSKoHE7y9UOw3zHZkG4s5LpjBpdDrfXVBNbeNhu8tRSiWQ+A/3zl+phkZe27bf6+a/r55Fe0eIW55eTlvHyKtRKeVM8R/umSUQbIODdXZX0qMT8tL4jyumsWJ7I//6vA4wppQaHn22uY94mRHdIdNH21tLLy6eNoZtDYf41ZsbyUn18eOLJ2N1QlJKqdiI/3DvuiPTDiiZbW8tx3HzvHE0HGjj0fe3kJuexM3zxttdklLKweI/3DMjwn0EExF+dNFk9h1s5Z43NjIqNYkrZ5faXZZSyqHiP9z9GeDPHHF93Xvicgn3XDGd/YfaueOF1WSlJHH+lJHZlKSUim/xf0EVwt0hR364AyR5XDz4lVlMK87i1mdWsGRzg90lKaUcyEHhPjL7uvckJcnDY9fPpjQnha89UcW62u7jsCml1NA4I9yzSuKiWSZSdmoST/6fOaT5PVz32Idsbzhkd0lKKQdxRrhnlkBrEwSa7K5kQMZkJfPHr86hPRjimkeXUN+id3FSSkWHQ8K92HqMs7N3gPH56Tx6/Wzqmlu5/rEPaQm0212SUsoBnBHuWeEuhXFyUbW7WaXZPPiVWWzc3cKNTy4j0B60uySlVJxzRriPsDsyDca8E/P59Ren88HmBr6r92FVSg1R/PdzB0jNA7cPGkfe0L8DcdnMIhoOtvHzV9dx/n8u4qKKQi6eVsiEgnS7S1NKxRlnhLvLZbW7x2mzTKSvfmosOalenv1wB/e/+xG/fecjJhakcVHFGC6aNprx+Rr0Sqm+OSPcwQr3OLyg2pPLZxZz+cxi6poDvLF2N6+u2sV/vrOJ+97exIkF6Vw0rZALKwoZn59md6lKqRFKrJsoDb/KykpTVVUVvQ2+fAt89DbcvjF62xxB6poDvL5mN6+t2sXSbfswBiaNTufCikIun1lESU6K3SUqpYaBiCwzxlT2tZ6DztxL4cBu6Gi17q3qMPkZfq47vZzrTi9nT3OA11fv4i+rd3Pf25v43bsf87VPj+WbZ48nJck5H6lSavCc0VsGIob+jd8eM/1VkOHn+jPG8tw3TuP9H5zN56aP4b//9gnn/ObvvLZqF3Z9G1NKjRzOCffOHzI54KLqQIzJSuY3X5rO8984jeyUJG55ejlX/2EJH+1psbs0pZSNHBTuEXdkSkCV5Tn8+dZP8fPLprK2tpnP/vY9fvHqOv3Fq1IJyjnhnlEESEI0y/TG7RKuObWMhbfP44uVJTzy/hbO/s3feXFFjTbVKJVgnBPuniRIL0y4Zpme5KQm8cvPV/DyLWdQlJXMdxes5Ev/8wFra+NrYDWl1OA5J9wh3Nc9vn+lGk3TirN44abTuecL09hcf5DP/ddifvjiaj6pP2B3aUqpGOsz3EXkURGpE5E1vbw+T0SaRKQ6PP0k+mX2U1b83JFpuLhcwpdml/DubfO45tQyFizdwTm/+TvzH/6AV1bW0tqhg5Qp5UT96RT9OPA74MnjrPOeMebiqFQ0FJklsO4VCIWsIQlUl8wUL3ddOpVbzh7P/1bV8MyH2/nWMyvISU3iiycXM39OKWNzU+0uUykVJX2GuzFmkYiUx76UKMgqgVA7HNgDGYV2VzMi5af7ueWs8dx05jje+3gvTy/Zxh8Wb+F/Fm3m9HGj+PIppZx30miSPHpwVCqeRevnjKeJyEqgFrjdGLO2p5VE5EbgRoDS0tIovXWEzIhx3TXcj8vlEs6cmMeZE/Ooaw7wXNUOnvlwB998egW5aUlccXIJV80poWyUns0rFY/6NbZM+Mz9VWPM1B5eywBCxpgDInIh8FtjzIS+thn1sWUA9qyDB0+DLzwCFVdEd9sJIBgyvPdRPU8v2c47G+oIhgzTizM5b8pozjupgPH5aYiI3WUqldCGbWwZY0xzxPxfROS/RSTXGLN3qNsesAQagiAW3C5h3on5zDsxn91NAV5YUcOba/fwqzc38qs3NzI2N5VzTyrgvJMKmFmajdulQa/USDXkcBeR0cAeY4wRkTlYPXAahlzZYPjSwZ+lPWaiYHSmn5vnjefmeePZ0xzgrXV7+Ou6PTz2/hYeXrSZ3LQkPjO5gHNPKuCM8bn4vW67S1ZKRegz3EXkGWAekCsiNcBPAS+AMeYh4ArgJhHpAA4D842dP4fMKknYIQhipSDDz1dOLeMrp5bRHGjnbxvr+Wt4nPlnl+4gJcnNmRPzuGDqaM49qUBHplRqBHDOeO6dnrkK9m+Fmz+I/rbVUVo7gvxz8z7+unY3b63bQ11LK8leN+dNKeCS6WP49IQ87XWjVJQl3njunTJLYMt7YAzoxb+Y8nncXT1ufn7pVD7cuo+Xq2t5fc0uXq6uJSvFy4UVhVwyfQxzynNwaRu9UsPGeeGeVQJtLRBoguQsu6tJGC6XcOoJozj1hFHcdckU3vuonpera3lx+U6eXrKdwkw/n5s+hkumj2HKmAztdaNUjDkv3DuH/m3aoeFukySPi3MmF3DO5AIOtXXw1ro9/HllLY8uti7GnpCXyvlTRjO9OJOpRZkUZSVr2CsVZc4N98YdMLrC3loUKUkeLp1RxKUzith/sI3X1+zmlZU7eXjRZoIh63pPTmoSU8ZkUFGUaU3FGvhKDZXzwj0r4sxdjSjZqUl8+ZRSvnxKKYH2IOt3NbNmZxOrdzaxemczDy/aTEc48LNTvEwNh/204izOGD+KdL/X5j1QKn44L9xT88Dj13Af4fxeNzNLs5lZmt21LNAeZMPuFlbvbGJNjRX6nYGf5HZx6rhRnHeS1be+IMNvY/VKjXzOC3eR8LjuGu7xxu91M6MkixklR66VBNqDrNzRyNvrrR9R/eilNfzopTVML87k3JMKOPek0Uws0GERlOrOef3cAZ68DFqb4evvxmb7yhbGGD6qO9D1a9mVOxoBKM1J6RoW4eSybDxu7VuvnCtx+7mDdea+6U27q1BRJiJMLEhnYkE6t5xlDYvw9vo9vLVuD3/8YBuPLN5CVoqXE3JTyU3zMSrNR25aUng+iVGpPvLSrcfMZK/2u1eO5sxwzyqFg3XQHgCvts06VUGGn6tPKePqU8o40NrBok31LNxQR23TYbY1HGL59v3sO9hGqIcvpx6XkJOaxImj0zl7Uj5nT8rX4Y2Vozgz3Du7QzbvhFHj7K1FDYs0n4cLKwq5sOLocfyDIcP+Q200HGij4UArew+2sbellYaDrdS3tLJ8eyN3/Xkdd/15HePyUjlncgFnT8rn5LJsvNq8o+KYM8O9sztk43YN9wTndgm5aT5y03xAeo/rbGs4yLsb6nh3Qx2Pv7+VhxdtJsPvYe7EPM6ZnM+ZE/PJSU065u/agyH2NAfY1RSgtvEwtY0BdjVZj42H2phalMnp40ZxygmjyEzWbpxqeDkz3DOLrUftDqn6oWxUKjecMZYbzhjLgdYOFn+0l3c37OHdDfW8umoXLoGZpdlUFGVS1xLoCvG6lla690dI93sYk5lMut/Ds0u38/g/tuISqCjO4vRxozhjXC6V5dk6RLKKOWeGe0YRiEu7Q6oBS/N5uGDqaC6YOppQyLCmtol31ltn9f9btYOCDD+FWX7mTsijMCuZMZn+ox7TfEf+S7V2BFmxvZF/fLyXf3zSwO8XbebBv31CktvFrLIszhiXy+njRzGtOAuv20UoZGg63M7+Q23WdNCabzzUzr5DbTSGlwWNIcPvJSPZQ7rfS4bfQ0ayt2tZht9LZvh5mt+jN1VJUM7sCglw70kw9ky4/MHYvYdSA3CgtYOlW/bxj0/28v7HDazbZd3ELCXJjc/jovFw+zHfBDp5XEJWShLZKV7cLqEl0EFzoJ2WQEef75vm8xxzAOjpoJDms7YtAgJdvx2w5sMT1osCZCR7Kcjwk5/u0+sTvWjtCNJwoI36FusaT/0B63FmaRafnpA3qG0mdldIsC6qarOMGkHSfB7OmpTPWZPyAdh3sI1/bm5gyeYGgsaQnZJkTaleslKSyAk/z0r1ku7z9PhDrWDIcKC1g+bD7TQH2mk+3EFLoJ3mgLWs6XB714GgOTy/qynAxj0tXev21JtooHLTkshP9zM6009Bho/8dD8FGX5GZ1rz2alJeF2C2yV4XC7cbsHjsibrgHLsvhlj6AgZ2jpCtHWEaA+GaO0I0RYMdS3rCIXwedxk+L2k+z2k+T2DOtC0B0M0hf+9OqdDrUE6QiFCxhAMQShkrHljCIUMwZAhaKzlbcEQ+w4eG+JNh9t7fL+b5o0bdLj3l4PDvRhqltpdhVK9yklN6rGHz0C4XUJmsnfQF2xDIcPBtg6aA1bQB0PmqG8PxoDBhB+twDXh5c2H29ndHGBPc4A9za3hxwCrappoOHjs9Yi+9sPtErwuwSViBXgwNKBtdPJ7XaSHw77zG0qaz0O630PI0BXezZFB3hYc+Bt1k+x1k5/hIy/Nx4T8NE4fN4q8NB956daUG54flZZ9/a9cAAAKSklEQVSEzxP7ay7ODfesElj3EoSC4NKLV0r1xOWScBB6geSobbc9GKK+5UjgNx6yrhV0BK2z8WAoZD12PTe0h0IEg9aZcZLbRZLHdeQxPHndLnwRy71uF4fbg7SED06djwdaOw9Y1vNdTQFaAu24xDoYZiR7KclJYWr4wNh9ykj2kubzdB10XAIukYjn4XkRXC7wul0j7iK5c8M9swRCHdCyGzKL7K5GqYTidbsYk5XMmKzoHTDUwDj3KkhWqfXYVGNvHUopZQPnhnumjuuulEpcDg738A+ZGrfbW4dSStnAueHuS4PkbD1zV0olJOeGO4T7umubu1Iq8Tg73LNKdQgCpVRC6jPcReRREakTkTW9vC4icr+IfCwiq0RkVvTLHKTMYqtZxqYhFpRSyi79OXN/HLjgOK9/FpgQnm4ERs5gLpkl0HYADu+3uxKllBpWfYa7MWYRsO84q1wKPGks/wSyRGTwv6eOps5x3bXdXSmVYKLR5l4ERDZs14SXHUNEbhSRKhGpqq+vj8Jb90H7uiulElQ0wr2nwaJ7bOQ2xjxsjKk0xlTm5cV2RDQAsstB3PDXH8GHv4fWlti/p1JKjQDRCPcaoCTieTFQG4XtDl1KDlz5/8CfBX+53Rrj/Y07YN9muytTSqmYika4vwJcG+41cyrQZIzZFYXtRsekC+HGhfC1d2Di+fDhw3D/LHh6PnyyUHvSKKUcqc87MYnIM8A8IBfYA/wU8AIYYx4Sa5T932H1qDkE3GCM6fMWSzG/E1NvmnfBsseg6lE4WA+5J8Ip/wLT50NS6vDXo5RSA9DfOzE59zZ7felohbUvwj8fhF3V4MuEWdfA9Ksg/yRwOfv3XUqp+KTh3l/GWHdsWvIQrHvZGgPenwWlp0HZ6VB2BhROA/fg7nSjlFLRpPdQ7S8RKJljTS274ZN3Yds/rGnT69Y63hTr9bIzrNAvrgSv3oRAKTVy6Zn78bTsge3hoN/2AexZAxhweaHoZCvks8utKavUmjT0lVIxpGfu0ZBeAFMutyawhjHYvuRI4H/4ewi2Hv03aQXhoC+D7LIj81mlkDIKfBnanq+UijkN94FIzoYTL7AmgFAIDuyBxm2wf5t1Y5DGrdZ8zVLrgq3pdld1cUNylrWt5GxIzjkynxIxn5QGSSngTQ0/pli9ebwp1rcD6em3Y0opZdFwHwqXCzIKran01GNfD3ZA804r9Jt2wKF9cHif9Q2gc2rZBXXrrfm2/v6CVsJhHw59bwp4fODx9/7o9VuPbh94ksCdZF0kdvus+a5lEVPXMl8P8z7r7/Ugo9SIpOEeS26P1TSTXda/9TvaINBoHQTaDkL7QWg7FPF4KLz8ELQfjpg/ZP1tR8CaAo1WV8+OwNGP7YfpZWSIIexjRNB7/Fbwe/zhg0PnwcXXbVnkOuHX3b4+5o9z4PL49CCjVDca7iOJJwnS8q0pFoyxunoG26ywD7Zb853TUctarQPGUa9FzofX7VzeuazzIBPsPNi0WQeryOedB5tgq/UYjQNOZNhHfrvoOkhEHFh6PZB0OyBFvhb5vPOg1PWo32bUyKPhnkhEwk0x3pHza1xjwgeUyANDOPS7DhydB4XDEd9EAr1/O4k8GEW+HmgMH7C6H4Raj70wPhS9HgS8vb/W9S2kp28m3ZZ7/eBJtq69dF6D6ZzXbzEqTMNd2Usk3EyTBD4b6zDm2G8XkQeHyNeCkQeeHr65HPPYw7K2A9b1l8jtRR6IQh2D3BE5OuyPmk85Mp+U0u31VOux80J+Uqo1742YT0qxDi568IgLGu5KQfggEz6DHgmCHVbot3d+S4kI/vbD1reY9s7pkLVe57WYrsfD1vWaztcCjdYF/PZD4Ws44ddNqP91iSsc9OHA96VHTBndnvewzJ8RXpZhXZNSMaP/ukqNRG6PNcW6+ayzWaz7RfvOqT1ivu2AtU7X/AHrHgmtLdYgfJ3zrc39O2B4U44+APg7DwSZVndhf1bPj8nZ4M/UIUH6oOGuVCKLbBZLzo7ONo2xvhVEhn1rszUfiJxvOvJ6ILysZbe1PNBkHWiOJyntOAeAzvnsHtZJjAODhrtSKrpEwu32KdavvAero9UK+cONVpNS5OPh/ccu27f5yPO+Dgze1COB7888EvqRB4Cjvil0/vAwa+Q03fVBw10pNTJ5fIPvGtx1YNh/7MGh64DRdOR54w4IrA5/m2g+/ra9KUeHfeS3g2OWZR9Z5ssc1qFHNNyVUs4zlANDsMMK+MhvB13z4YNF1wFjP+zbcuT1435jkCPfFmZ/DU7/5qB3rz803JVSKpLbY43zlJIz8L/taD1yMDjqgLD/6INE2hCaq/pJw10ppaLF47OuMwzlWkOU6NizSinlQBruSinlQBruSinlQBruSinlQBruSinlQBruSinlQBruSinlQBruSinlQGJMlO+p2d83FqkHtg3yz3OBvVEsZyRy+j46ff/A+fuo+2ePMmNMXl8r2RbuQyEiVcaYSrvriCWn76PT9w+cv4+6fyObNssopZQDabgrpZQDxWu4P2x3AcPA6fvo9P0D5++j7t8IFpdt7koppY4vXs/clVJKHUfchbuIXCAiG0XkYxH5N7vriTYR2Soiq0WkWkSq7K4nGkTkURGpE5E1EctyROQtEfko/BiluzMPv172704R2Rn+HKtF5EI7axwKESkRkYUisl5E1orIt8PLnfQZ9raPcfs5xlWzjIi4gU3AuUANsBS4yhizztbCokhEtgKVxpiR2L92UERkLnAAeNIYMzW87B5gnzHm38MH6WxjzA/srHOwetm/O4EDxphf21lbNIhIIVBojFkuIunAMuAy4Hqc8xn2to9fIk4/x3g7c58DfGyM2WyMaQOeBS61uSbVB2PMImBft8WXAk+E55/A+o8Ul3rZP8cwxuwyxiwPz7cA64EinPUZ9raPcSvewr0I2BHxvIY4/wB6YIC/isgyEbnR7mJiqMAYswus/1jAIO5kPOJ9U0RWhZtt4rbJIpKIlAMzgSU49DPsto8Qp59jvIW79LAsftqV+ucMY8ws4LPALeGv/Cr+PAiMA2YAu4Df2FvO0IlIGvAn4DvGmGa764mFHvYxbj/HeAv3GqAk4nkxUGtTLTFhjKkNP9YBL2I1RTnRnnA7Z2d7Z53N9USVMWaPMSZojAkBvyfOP0cR8WKF3lPGmBfCix31Gfa0j/H8OcZbuC8FJojIWBFJAuYDr9hcU9SISGr4Yg4ikgqcB6w5/l/FrVeA68Lz1wEv21hL1HWGXtjlxPHnKCICPAKsN8bcG/GSYz7D3vYxnj/HuOotAxDuivSfgBt41Bhzt80lRY2InIB1tg7gAZ52wv6JyDPAPKxR9vYAPwVeAp4DSoHtwBeNMXF5UbKX/ZuH9VXeAFuBf+lsn443IvIp4D1gNRAKL/6/WG3STvkMe9vHq4jTzzHuwl0ppVTf4q1ZRimlVD9ouCullANpuCullANpuCullANpuCullANpuCullANpuCullANpuCullAP9f2MqwMQS9fWOAAAAAElFTkSuQmCC%0A)

```python
weights = deep_fm_model.get_weights()
fm_1_weight, fm_2d_weigth, deep_weight = weights[-2]
print("""
contribution of different part of model
    weight of 1st order fm: %5.3f
    weight of 2nd order fm: %5.3f
    weight of dnn part: %5.3f
""" % (fm_1_weight, fm_2d_weigth, deep_weight))
```
```

contribution of different part of model
    weight of 1st order fm: -0.883
    weight of 2nd order fm: 1.518
    weight of dnn part: 0.469
```
