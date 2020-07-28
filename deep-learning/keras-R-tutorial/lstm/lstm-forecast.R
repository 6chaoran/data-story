library(dplyr)
library(data.table)

df <- fread('data/rainfall-monthly-total.csv')

df <- df %>%
  mutate(month_index = rank(month)) %>%
  mutate(label = total_rainfall/800)

shift_i <- function(df,i){
   res <- df %>%
    inner_join(df %>% 
                 transmute(month_index = month_index+i,
                           feature = label), 
               by = 'month_index')
  colnames(res)[ncol(res)] <- paste0('shifted_',i)
  return(res)
}

dim_t <- 12
for(i in 1:dim_t){
  df <- shift_i(df,i)
}

#==== LSTM =====
library(keras)
model.lstm <- function(dim_t, k, cnn, deep){
  inputs <- layer_input(shape = c(dim_t,1), name = 'input_layer')

  lstm <- inputs %>%
    layer_lstm(units = k, go_backwards = T, name = 'lstm_layer')
  
  all_in_use <- c(inputs %>% layer_flatten(),
                  lstm)
  if(cnn){
    cnn <- inputs %>%
      layer_conv_1d(filters = 10, kernel_size = 3, padding = 'same')
    all_in_use <- c(all_in_use, cnn %>% layer_flatten())
  }
  
  if(deep){
    outputs <- all_in_use %>%
      layer_concatenate(name = 'concat_layer') %>%
      layer_dropout(0.5) %>%
      layer_dense(128, activation = 'relu') %>%
      layer_dropout(0.5) %>%
      layer_dense(1, activation = 'linear',name = 'dense_layer')
  } else {
    outputs <- all_in_use %>%
      layer_concatenate(name = 'concat_layer') %>%
      layer_dense(1, activation = 'linear',name = 'dense_layer')
  }

  model <- keras_model(inputs, outputs)
  model %>% compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mse')
}

sampling_generator <- function(df, batch_size) {
  function() {
    df.sample <- df %>% sample_n(batch_size)
    X_sample <- as.matrix(df.sample %>% select(starts_with('shifted_')))
    t <- ncol(X_sample)
    X_sample <- keras_array(X_sample)
    X_sample <- array_reshape(X_sample, c(-1,t,1))
    Y_sample <- (df.sample$label)
    list(X_sample, Y_sample)
  }
}

model1 <- model.lstm(12, 10, cnn = T, deep = T)

callbacks <- list(
  callback_csv_logger(format(Sys.time(),'model/log/lstm_model.csv')),
  callback_early_stopping(patience = 5),
  callback_tensorboard(format(Sys.time(),'model/tensorboard/lstm_model_%Y%m%d_%H%M')))

hist1 <- model1 %>%
  fit_generator(
    sampling_generator(df ,batch_size = 8),
    callbacks = callbacks,
    steps_per_epoch = 10,
    epochs = 50,
    verbose = 1,
    validation_data = sampling_generator(df, batch_size = 32),
    validation_steps = 1)

df.test <- sampling_generator(df,360)()[[1]]
pred <- model1 %>% predict(df.test)
df$pred.lstm <- pred[,1]
with(df, rmse(label, pred.lstm))
#==== LightGBM =====
library(xgboost)
xgb.data <- xgb.DMatrix(data = as.matrix(df %>% select(starts_with('shifted_'))),
                        label = df$label)
grid = list(objective = "reg:linear",
            eval_metric  = "rmse",
            max_depth  = 6,
            eta =0.1,
            colsample_bytree  = 1,
            subsample  = 1)
model.xgb <- xgb.train(params = grid, 
                        data = xgb.data, 
                        nrounds = 100)
pred <- predict(model.xgb, xgb.data)

df$pred.xgb <- pred


#==== Inference ====
library(Metrics)
with(df, rmse(label, pred.xgb))
with(df, rmse(label, pred.lstm))
df %>%
  select(month_index,label, pred.xgb, pred.lstm) %>%
  melt(id.vars = 'month_index') %>%
  ggplot(aes( x= month_index, y = value, color = variable)) +geom_line()

library(ggplot2)
df %>%
  select(month_index, pred, label) %>%
  melt(id.vars = 'month_index') %>% 
  ggplot()+geom_line(aes(x = month_index, y = value,color = variable, alpha = 0.5))


df$month <- NULL

xgb.inference <- function(df, n_iter){
  for(i in 1:n_iter){
    df.test <- xgb.DMatrix(as.matrix(df %>% 
                                       arrange(desc(month_index)) %>% 
                                       head(12) %>%
                                       select(total_rainfall) %>%
                                       t()))
    pred <- predict(model.xgb, df.test)
    df <- df %>% rbind(data.frame(month_index = max(df$month_index)+1,
                           total_rainfall = pred))
  }
  
  return(df)
}

df_ <- xgb.inference(df, 12)
df_$type <- NA
df_$type[1:438] <- 'historical'
df_$type[439:450] <- 'pred'

df_ %>%
  ggplot(aes(x = month_index, y = total_rainfall, color = type))+geom_line()
