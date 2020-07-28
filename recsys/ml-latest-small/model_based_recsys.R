setwd("~/Desktop/github/RecommenderSystem/ml-latest-small")

rm(list = ls())
gc()

library(data.table)
library(dplyr)
library(tidyr)
library(reshape2)
library(Metrics)

schema <- c("userId" = "integer",
            "movieId" = "integer",
            "rating" = "numeric",
            "timestamp" = "character")
df <- fread("ratings.csv", colClasses = schema)

## label train (+1) / val (-1)
df$label <- 1
set.seed(2333)
df$label[sample(1:nrow(df),0.3*nrow(df))] <- (-1)

# Y = U X I
Y = dcast(df, userId ~ movieId, value.var = 'rating', fill = 0)
Y <- as.matrix(Y[2:ncol(Y)])

R <- dcast(df,userId ~ movieId, value.var = 'label', fill = 0)
R <- as.matrix(R[2:ncol(R)])
R.train <- R
R.train[R.train<1] <- 0
R.val <- (-R)
R.val[R.val<1] <- 0

## loss function - minimal square error
loss <- function(Y,U,I,R,lambda){
  return((sum(R*(Y-U %*% t(I))**2) + lambda*sum(U**2) + lambda*sum(I**2))/sum(R)/2)
}

## gradient function - user matrix
grad_u <- function(Y,U,I,R,lambda) (-(R*(Y - U%*%t(I))) %*% I + lambda*abs(U))/sum(R)

## gradient function - item matrix
grad_i <- function(Y,U,I,R,lambda) (-t(R*(Y - U%*%t(I))) %*% U + lambda*abs(I))/sum(R)

## gradient descent optimizer
gd <- function(Y,U,I,R, lambda, alpha, maxIter,thresh = 1e-4){
  loss0 <- loss(Y,U,I,R[[1]],lambda)
  alpha0 <- alpha
  pred0 <- (U%*%t(I))
  rmse.train0 <- rmse(pred0[R[[1]]>0],Y[R[[1]]>0])
  rmse.val0 <- rmse(pred0[R[[2]]>0],Y[R[[2]]>0])
  gr_u <- grad_u(Y,U,I,R[[1]],lambda)
  gr_i <- grad_i(Y,U,I,R[[1]],lambda)
  
  rmse.train.history <- NULL
  rmse.val.history <- NULL
  
  for(i in 1:maxIter){
    ## update user,item matrix simutaneously
    U <- U - gr_u * alpha
    I <- I - gr_i * alpha
    
    loss1 <- loss(Y,U,I,R[[1]],lambda)
    
    ## check the train/validation error
    pred <- (U%*%t(I))
    rmse.train <- rmse(pred[R[[1]]>0],Y[R[[1]]>0])
    rmse.val <- rmse(pred[R[[2]]>0],Y[R[[2]]>0])
    
    ## record rmse history
    rmse.train.history <- c(rmse.train.history, rmse.train)
    rmse.val.history <- c(rmse.val.history,rmse.val)
    
    ## stop citeria: increasing loss, validation rmse 
    ## or the difference of loss is less than treshold
    if (abs(loss1-loss0)<thresh | loss1 > loss0 | rmse.val > rmse.val0) break
    cat('iter',i,': loss',loss1,'alpha',alpha,'rmse.train',rmse.train,
        'rmse.val',rmse.val,'\n')
    
    # update the starting point
    loss0 <- loss1
    pred0 <- pred
    rmse.train0 <- rmse.train
    rmse.val0 <- rmse.val
    
    ## update gradients
    gr_u <- grad_u(Y,U,I,R[[1]],lambda)
    gr_i <- grad_i(Y,U,I,R[[1]],lambda)
  }
  
  rmse.history <- data.frame(iter = 1:i, rmse.train = rmse.train.history, rmse.val = rmse.val.history)
  
  return(list(U = U,I = I,loss = loss1,rmse.history = rmse.history))
}

## alternate least square optimizer
als <- function(Y,U,I,R,lambda, alpha, maxIter,thresh = 1e-4){
  loss0 <- loss(Y,U,I,R[[1]],lambda)
  alpha0 <- alpha
  pred0 <- (U%*%t(I))
  rmse.train0 <- rmse(pred0[R[[1]]>0],Y[R[[1]]>0])
  rmse.val0 <- rmse(pred0[R[[2]]>0],Y[R[[2]]>0])
  gr_u <- grad_u(Y,U,I,R[[1]],lambda)
  gr_i <- grad_i(Y,U,I,R[[1]],lambda)
  
  # rmse history
  rmse.train.history <- NULL
  rmse.val.history <- NULL
  iter <- 0
  for(i in 1:maxIter){
    
    # update User Matrix
    U <- U - gr_u * alpha
    loss1 <- loss(Y,U,I,R[[1]],lambda)
    pred <- (U%*%t(I))
    rmse.train <- rmse(pred[R[[1]]>0],Y[R[[1]]>0])
    rmse.val <- rmse(pred[R[[2]]>0],Y[R[[2]]>0])
    if (abs(loss1-loss0)<thresh | loss1 > loss0 | rmse.val > rmse.val0) break
    cat('U iter',i,': loss',loss1,'alpha',alpha,'rmse.train',rmse.train,
        'rmse.val',rmse.val,'\n')    
    
    # record rmse history
    rmse.train.history <- c(rmse.train.history, rmse.train)
    rmse.val.history <- c(rmse.val.history, rmse.val)
    iter <- iter + 1
    
    # update the gradient
    gr_u <- grad_u(Y,U,I,R[[1]],lambda)
    
    # update the starting point
    loss0 <- loss1
    pred0 <- pred
    rmse.train0 <- rmse.train
    rmse.val0 <- rmse.val
    
    # update Item Matrix
    I <- I - gr_i * alpha
    loss1 <- loss(Y,U,I,R[[1]],lambda)
    pred <- (U%*%t(I))
    rmse.train <- rmse(pred[R[[1]]>0],Y[R[[1]]>0])
    rmse.val <- rmse(pred[R[[2]]>0],Y[R[[2]]>0])
    if (abs(loss1-loss0)<thresh | loss1 > loss0 | rmse.val > rmse.val0) break
    cat('I iter',i,': loss',loss1,'alpha',alpha,'rmse.train',rmse.train,
        'rmse.val',rmse.val,'\n')
    
    # record rmse history
    rmse.train.history <- c(rmse.train.history, rmse.train)
    rmse.val.history <- c(rmse.val.history, rmse.val)
    iter <- iter + 1
    
    # update the gradient
    gr_i <- grad_i(Y,U,I,R[[1]],lambda)
    
    # update the starting point
    loss0 <- loss1
    pred0 <- pred
    rmse.train0 <- rmse.train
    rmse.val0 <- rmse.val
  }
  
  rmse.history <- data.frame(iter = 1:iter, rmse.train = rmse.train.history, rmse.val = rmse.val.history)
  
  return(list(U = U,I = I,loss = loss1,rmse.history = rmse.history))
}


## initiaize U and I
m <- nrow(Y)
n <- ncol(Y)
k <- 5
set.seed(2333)
U = matrix(runif(m*k),m,k)
I = matrix(runif(n*k),n,k)

## train with models
gd.time <- system.time(res.gd <- gd(Y,U,I,list(R.train,R.val),0.3,30,1000)) # 70s, rmse:0.946
als.time <- system.time(res.als <- als(Y,U,I,list(R.train,R.val),0.3,30,500)) #46s,rmse:0.994

## save model result
save(gd.time,als.time,res.gd, file = 'model_result')

## plot the rmse hisotry
library(ggplot2)
gd <- melt(res.gd$rmse.history, id.vars = 'iter')
als <- melt(res.als$rmse.history, id.vars = "iter")
gd$method = 'gradient descent'
als$method = 'alternating least square'
history <- rbind(gd,als)
history$method <- paste(history$method, history$variable)
hist.plot <- ggplot(history,aes(x = iter, y = value,colour = factor(method)))+geom_point(alpha = 0.2)+geom_line()+
  labs(y = 'rmse')

