---
title: "Implementation of Model Based Recommendation System in R"
date: 2016-12-19 16:16:01 +0800
categories: recsys
tags: R
classes: wide
---

The most straight forward recommendation system are either user based CF (collaborative filtering) or item based CF, which are categorized as memory based methods. User-Based CF is to recommend products based on behaviour of similar users, and the Item-Based CF is to recommend similar products from products that user purchased. No matter which method is used, the user-user or item-item similarity matrix, which could be sizable, is required to compute.   

While on the contrast, a model based approach could refer to converting recommendation problem to regression, classification, learning to rank problem. Matrix Factorization, which is also known as latent factor model,SVD, is one of the most commonly used model based methods. In this post, a variety methods of CF will be discussed, including:

* Gradient Descent CF (GD)
* Alternating Least Square (ALS)

### 1. dataset

This post is demonstrated on MovieLens dataset, which consists of 10M rows of user-movie rating records.

### 2. formula

The matrix factorization representation: the rating matrix could be considered as the cross product of two matrix.

```
Y = U x M
```

where `Y` is the rating matrix with dimension (u,p),
`U` is the decomposed user matrix,
`M` is the decomposed movie matrix.

loss function:

![image](https://6chaoran.files.wordpress.com/2016/12/1.jpg)

gradient function:

![image](https://6chaoran.files.wordpress.com/2016/12/2.jpg)

### 3. model optimizer

There are two methods that we are going to compare:

* Gradient Descent CF: update the gradients of user and item matrix simultaneously
* Alternating Least Square: update the gradients of user and item matrix alternatively

Ideally, gradient descent should have better model performance; while alternating least square could have faster convergence. When the size of data grows large, the difference will be more severe.

### 4. result comparison

```
method  iter    rmse.train  rmse.val    time
GD      205     0.8047878   0.9328183   153.463
ALS     280     0.8295948   0.9366344   153.357
```

![image](https://6chaoran.files.wordpress.com/2016/12/3.png?w=700)

### R code
[R code in github](https://github.com/6chaoran/DataStory/blob/master/RecommenderSystem/ml-latest-small/model_based_recsys.R)