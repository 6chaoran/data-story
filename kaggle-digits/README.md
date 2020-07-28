<h1>Recognize the Digits</h1>

This time I am going to demostrate the kaggle 101 level competition - <a href="https://www.kaggle.com/c/digit-recognizer" target="_blank">digit recogniser</a>.
<h2>Mission:</h2>
We are asked to train a model to recogize the digit from the pixel data in this competition. <a href="https://www.kaggle.com/c/digit-recognizer/data" target="_blank">The data set</a> is available here.
description of the data:
1. label: the integers from 0 - 9;
2. features: pixel001-pixel784, which are rolled out from 28x28 digit image;
3. pixel data is ranged from 0 -255, which indicating the brightness of the pixel in grey scale;
<h2>Visualize the digit:</h2>
Let's randomly look at 100 digit examples:

[code lang="r"]display(test[sample(28000,100),],28)[/code]

[caption id="attachment_81" align="alignnone" width="504"]<img class="wp-image-81 size-full" src="https://6chaoran.files.wordpress.com/2015/07/unnamed-chunk-1-1.png" alt="unnamed-chunk-1-1" width="504" height="504" /> 28x28 visualization[/caption]

<!--more-->
<h2>Dimension Reduction 1:</h2>
As we are having 784 features, which are prabably too many for training. We noticed the digits are well distinguishable, so that may be managable with lower resolution, say 28x28 to 14x14, which will significantly reduces the features from 784 to 196!
The idea is to find the brightest pixel (max) within the adjance 2x2 grid.

[code lang="r"]reduceDim&lt;-function(data){
  pos&lt;-matrix(1:784,28,28,byrow=T)
  offset&lt;-seq(1,28,2)
  n=0
  train.reduced&lt;-data.frame(index=1:nrow(data))
  if(!is.null(data$label)) train.reduced$label&lt;-data$label
  data$label&lt;-NULL
  for (i in offset){
    for (j in offset){
      px&lt;-as.numeric(pos[i:(i+1),j:(j+1)])
      px&lt;-apply(data[,px],1,max)
      index&lt;-paste0('px',n)
      n=n+1
      train.reduced[index]&lt;-px
    }
  }
  train.reduced$index&lt;-NULL
  return (train.reduced)
}
train.reduced=reduceDim(train)
test.reduced=reduceDim(test)[/code]

Let's take a look at the digit images after dimension reduction.

[code lang="r"]display(test.reduced[sample(28000,100),],14)[/code]

[caption id="attachment_82" align="alignnone" width="504"]<img class="wp-image-82 size-full" src="https://6chaoran.files.wordpress.com/2015/07/unnamed-chunk-3-1.png" alt="14x14 visualization" width="504" height="504" /> 14x14 visualization[/caption]

The digit is still well recognizable!
<h2>Dimension Reduction 2:</h2>
Besides the manual dimension reduction done earlier, we have a smarter alogrithm call 'Principle Component Analysis' (PCA).
PCA is a method to compress the data and projected to n component axis. This comression and recovery process will incur some information loss, which is expressed the variance retained. In this case, we set the variance retrained to be 90%.

[code lang="r"]library(caret)
pca&lt;-preProcess(rbind(train.reduced,test.reduced),method='pca',thresh=0.9)
train.pca&lt;-predict(pca,train.reduced)
test.pca&lt;-predict(pca,test.reduced)[/code]

<pre><code>## 
## Call:
## preProcess.default(x = rbind(train.reduced, test.reduced), method =
##  "pca", thresh = 0.9)
## 
## Created from 70000 samples and 101 variables
## Pre-processing: principal component signal extraction, scaled, centered 
## 
## PCA needed 47 components to capture 90 percent of the variance.</code></pre>
With PCA implemented, we reduced the number of features to 47!
<h2>Train with Linear SVM:</h2>
For illustration purpose, we only trained 500 data points.

[code lang="r"]ctrl&lt;-trainControl(method='cv',number = 10)
inTrain=sample(42000,500)
run_time&lt;-system.time(fit&lt;-train(factor(label[inTrain])~.,data=train.pca[inTrain,],
            trControl = ctrl,
            method='svmLinear'))
print (fit)[/code]

<pre><code>## Support Vector Machines with Linear Kernel 
## 
## 500 samples
##  46 predictor
##  10 classes: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 449, 452, 449, 449, 451, 450, ... 
## 
## Resampling results
## 
##   Accuracy   Kappa     Accuracy SD  Kappa SD  
##   0.8219855  0.801539  0.03704343   0.04130649
## 
## Tuning parameter 'C' was held constant at a value of 1
## 
</code></pre>
<h2>Summary:</h2>
Simple linear SVM is giving fairely good accuracy with only small part of the entire training data.
Further Explore Area:
1. Increase PCA threshold
2. Using higher order SVM / Gaussian Kernel SVM or Neural Network/Random Forest
3. Train with more data

The completed R code is available <a href="https://github.com/6chaoran/kaggle/blob/master/digit-recognizer/digit-recognize.R" target="_blank">here</a>.
