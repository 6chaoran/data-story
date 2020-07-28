# Review on Stanford Machine Learning Course
## Overview

I have been signed up for this course for a long time. And since last week, I finally managed to complete it with a good score.

![image](https://6chaoran.files.wordpress.com/2015/07/ml.jpg)

This course is taught by Andrew Ng, who is also the co-founder of Cousera. The content of the course spans from supervised learning to unsupervised learning, as well as some advice for the ML model improvement, special topics on pipeline setup and implementation on large scale data.

I personally feel this course is very beneficial for the beginners like me. The fun, but probably useless part is actually to understand how those well-established algorithms work. Still remember at the end of video lecture of logistic regression, I was surprised to hear Mr. Ng said, "Probably you are now better than most of the Engineers work in Silicon Valley". I guess what he was trying to say is that some of the Engineers don't even brother to understand the theories before they implementing the algorithms. After that I read some blogs saying, you actually don't need a Coursera course to do machine learning, then they shown a series of screenshots of the software to implement a ML algorithm, and I feel that is just turning human into machine.

## Recall the gist
### Optimization:
Optimization is a very important concept throughout this course. Prof. Ng formulate each machine learning algorithm into an optimisation problem. 

#### Method: 
Gradient descent method, which is a numerical method to search for the minimal,  is used throughout the course. However Prof. Ng mentioned self-coded gradient descent is not recommended, so a more complex Matlab function **fminunc** is used instead.  In the large-scale problem, stochastic gradient descent is used instead to get faster learning.
#### Problem-defining: 
Each optimizaiton need to be provided with cost function (optimisation objective) and gradient function to implement gradient descent. 
In ordinary linear regression, the cost function is simply the sum of square errors. 
In logistic regression, the cost function is similar to linear regression except the estimation is transformed by sigmoid function to bound to interval [0,1]. 
While in neural networks, the cost function is similar to logistic regression but with multiple layers, so the understanding of the cost function may not be intuitive.

### Regularization:
In this course, the concept of model bias-variance trade-off is highly emphasised. 
Regularization is added to model to give penalty for complicated model, which are either having more variables and higher coefficients. The regulariser used in this course is L2 - regularizor. The other common regularisers  are AIC/BIC, lasso (L1), elastic-net (combination of L1-L2)
In order to prevent overfitting, what suggested in the course is to plot the learning curve (train, test error over number of data used). If train data error is reducing while test data is not, the model is likely to have suffered overfitting. A worthwhile attempt is to increase regularisation of the model. 

### Supervised Learning:

Supervised learning is the machine learning with labelled response, which requires some manual work of labelling. 
Under the umbrella of supervised learning, there are two main categories: Classification (for discrete response) and Regression(for continuous response).
The supervised learning algorithms covered in this course:

#### Linear regression:

* Cost function: 	
![image](https://6chaoran.files.wordpress.com/2015/08/linear-regression1.jpg) 
* h(x) is simple the linear function.
* generally used for regression problem.

#### Logistic regression:
* Cost function: 
![image](https://6chaoran.files.wordpress.com/2015/08/logistic-regression.jpg)
*  h(x) is the sigmoid function
*  generally used for binary classification problem.

#### Neural Network:
* Cost function:
![image](https://6chaoran.files.wordpress.com/2015/08/neural-network.jpg)
* widely used for classification and regression problem.
* cost function assembles logistic regression, but with multiple output nodes instead of single node for logistic regression.
* forward propagation is to calculate the cost function and backward propagation is to calculate the gradient function of neural network.
* multiple hidden layers with more nodes are called deep learning, which can handle with complex problem but in the contrast is slow to train.

#### Support Vector Machine
* Cost function:
![image](https://6chaoran.files.wordpress.com/2015/08/svm.jpg)
* new parameter C is introduced in SVM, but it is similar as lambda in other cost function, serving as regularisation factor.
* cost1 represents the cost function of logistic regression when y=1, and similarly applies for cost0
* f denotes the kernel functions. Two common kernels are linear kernel and Gaussian kernel.
* linear kernel is very similar to logistic regression in terms of both theory and performance.
* additional parameter σ2 (variance) need to be specified when using Gaussian kernel to determine the Gaussian distribution.
* features scaling is required before implement Gaussian SVM.
* when # of features >> # of data: use logistic regression/linear kernel SVM, as gaussian or high order kernel SVM tends to overfit the model
* when # of features << # of data: use logistic regression/linear SVM or add more features.  Gaussian kernel SVM is slow to run.
* Only when # of feature is small (1-1,000) and # of data is moderate (10-10,000) , Gaussian kernel is good choice to model non-linear problem.
 
### Unsupervised Learning:
Unsupervised learning doesn't required response to be labelled, instead the model is learning by itself to find similar or correlated data to form a cluster.
#### k-means
* generally used for market segmentation, clustering.
* k-means involves three main steps: 
	1. randomly allocate the centroids 
	2. assign the data to nearest centroids 
	3. calculate the mean of each cluster and move the centroid to the newly calculated position. 
* The process is iterated until the sum of distance from each data to its centroid is minimised.

#### Principle Component Analysis
* PCA represent the high-dimensional data by first few eign-vectors. 
* generally used for speed-up model training, reduce storage space and visualizaton (high dimensional to 2D)
* mean-normalisation and feature scaling is required before implementing PCA
* PCA compression will incur information loss, which is measured by the variance retrained.
* either percentage of variance retrained or number of components need to be specified for PCA methods.

#### Anomaly Detection
* detect data that fall in the tails of Gaussian distribution.
* generally used for fraud detection in financial industrial and quality control in manufacturing.
* the general procedure is:
	1. choose the features
	2. fit to gaussian distribution parameters
	3. calculate the probability p(x), compare with threshold value.
* large number of negative data compared with very few positive data. precision/recall are no longer good metrics to evaluate model performance, F-score could be used instead. (F score is 2 * precision*recall/(recall+precision)))
* need to examine non-gaussian feature to be transformed to normal-like feature.
* independence of features is assumed for independent gaussian anomaly detection, while multivariate gaussian model is suitable for correlated features, which is much slower to run.
