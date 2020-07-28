library(keras)
# donwnload dataset
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

# compile model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy'))

my_generator <- function(x,y,batch_size) {
  function() {
    inSample <- sample(1:nrow(x), batch_size)
    x.sample <- x[inSample, ,]
    x.sample <- array_reshape(x.sample, c(length(inSample), 784))
    y.sample <- to_categorical(y[inSample],10)
    return(list(x.sample, y.sample))
  }
}

callbacks <- list(
  callback_csv_logger(format(Sys.time(),'model/log/mnist_%Y%m%d_%H%M.csv')),
  callback_early_stopping(patience = 5),
  callback_tensorboard(format(Sys.time(),'model/tensorboard/mnist_%Y%m%d_%H%M')))

valid.set <- my_generator(x_test,y_test,batch_size = nrow(x_test))()

hist <- model %>%
  fit_generator(
    my_generator(x_train,y_train,batch_size = 128),
    callbacks = callbacks,
    steps_per_epoch = 500,
    epochs = 50,
    verbose = 1,
    validation_data = valid.set,
    validation_steps = 1)
