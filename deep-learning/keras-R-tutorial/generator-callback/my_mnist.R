library(keras)
# donwnload dataset
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# define model

my_keras_model <- function(){
  
  inputs <- layer_input(shape = c(784), name = 'input_layer')
  outputs <- inputs %>%
    layer_dense(units = 256, activation = 'relu', name = 'fc_1') %>% 
    layer_dropout(rate = 0.4, name = 'dropout_1') %>% 
    layer_dense(units = 128, activation = 'relu', name = 'fc_2') %>%
    layer_dropout(rate = 0.3, name = 'dropout_2') %>%
    layer_dense(units = 10, activation = 'softmax', name = 'output_layer')
  model <- keras_model(inputs, outputs)
  
  # compile model
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy'))
}

# sampling generator
sampling_generator <- function(x,y,batch_size) {
  function() {
    inSample <- sample(1:nrow(x), batch_size)
    x.sample <- x[inSample, ,] / 255
    x.sample <- array_reshape(x.sample, c(length(inSample), 784))
    y.sample <- to_categorical(y[inSample],10)
    return(list(x.sample, y.sample))
  }
}

# sequential generator
seq_generator <- function(x,y,batch_size) {
  i <- 1
  function() {
    
    if(i+batch_size-1 > nrow(x)){
      inSample <- i:nrow(x)
      i <- 1
    } else {
      inSample <- i:(i+batch_size-1)
      i <- i+batch_size
    }
    
    x.sample <- x[inSample, ,]
    x.sample <- array_reshape(x.sample, c(length(inSample), 784))
    y.sample <- to_categorical(y[inSample],10)
    list(x.sample, y.sample)
  }
}

if(!file.exists(format(Sys.time(),'model/log/mnist_%Y%m%d_%H%M.csv'))){
  file.create(format(Sys.time(),'model/log/mnist_%Y%m%d_%H%M.csv'))
} 

callbacks <- list(
  callback_csv_logger(format(Sys.time(),'model/log/my_mnist_%Y%m%d_%H%M.csv')),
  callback_early_stopping(patience = 5),
  callback_tensorboard(format(Sys.time(),'model/tensorboard/my_mnist_%Y%m%d_%H%M')))

valid.set <- sampling_generator(x_test,y_test,batch_size = nrow(x_test))()

model <- my_keras_model()

hist <- model %>%
  fit_generator(
    sampling_generator(x_train,y_train,batch_size = 128),
    callbacks = callbacks,
    steps_per_epoch = 400,
    epochs = 30,
    verbose = 1,
    validation_data =valid.set,
    validation_steps = 1)