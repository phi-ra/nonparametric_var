get_inference_params <- function(model_name, layer_name, layer_n_out,
                                 data_x_nl,data_x_l, data_y) {
  last_layer <- keras_model(inputs = model_name$input,
                            outputs = get_layer(model_name, layer_name)$output)
  df_layer <- as.data.frame(predict(last_layer, list(data_x_nl, data_x_l)))
  df_layer$y <- data_y
  
  lm_t <- lm(y ~ ., data=df_layer)

  nas_lm <- unname(which(is.na(lm_t$coefficients)))
  vec_drop <- nas_lm - 1
  if (length(vec_drop)==0){
    vec_keep <- c(1:layer_n_out)
  } else {
    vec_keep <- c(1:layer_n_out)[-vec_drop]
  }

  z_mat <- as.matrix(df_layer[,vec_keep])
  x_mat <- as.matrix(df_layer[,(layer_n_out+1):(dim(df_layer)[2]-1)])
  y_mat <- as.matrix(df_layer$y)
  
  y_on_z <- gls(y_mat ~ 0 + z_mat)
  y_wiggle <- (y_mat- predict(y_on_z))
  
  x_on_z <- gls(x_mat ~ 0 + z_mat)
  x_wiggle <- (x_mat - predict(x_on_z))
  
  final_reg <- gls(y_wiggle ~ 0 + x_wiggle)
  
  return(final_reg)
  
}


get_keras_data <- function(input_data, train,
                           position_linear, position_nonlinear, position_y,
                           skip_first=0) {
  test <- train+1
  dim_nl <- length(position_nonlinear)
  dim_l <- length(position_linear)
  dim_y <- length(position_y)
  both_positions <- append(position_linear, position_nonlinear)
  dim_both <- length(both_positions)
  
  x_train_l <- array_reshape(as.matrix(input_data[(1+skip_first):train,
                                                  position_linear]), c(train-skip_first, dim_l))
  
  x_train_nl <- array_reshape(as.matrix(input_data[(1+skip_first):train,
                                                   position_nonlinear]), c(train-skip_first, dim_nl))
  
  x_train_both <- array_reshape(as.matrix(input_data[(1+skip_first):train,
                                                     both_positions]), c(train- skip_first, dim_both))
  
  y_train <- array_reshape(as.matrix(input_data[(1+skip_first):train,
                                                position_y]), c(train-skip_first, dim_y))
  
  ## Test
  x_test_l <- array_reshape(as.matrix(input_data[test:dim(input_data)[1],
                                                 position_linear]), c(dim(input_data)[1]-test+1, dim_l))
  
  x_test_nl <- array_reshape(as.matrix(input_data[test:dim(input_data)[1],
                                                  position_nonlinear]), c(dim(input_data)[1]-test+1, dim_nl))
  
  x_test_both <- array_reshape(as.matrix(input_data[test:dim(input_data)[1],
                                                    both_positions]), c(dim(input_data)[1]-test+1, dim_both))
  
  y_test <- array_reshape(as.matrix(input_data[test:dim(input_data)[1],
                                               position_y]), c(dim(input_data)[1]-test+1, dim_y))
  
  ## Pack together
  list_train <- list(y_train, x_train_l, x_train_nl, x_train_both)
  list_test <- list(y_test, x_test_l, x_test_nl, x_test_both)
  
  final_list <- list(list_train, list_test)
  return(final_list)
  
}

exceeded_estimation <- function(VaR_estimate, true_returns, alpha) {
  differences <- true_returns - VaR_estimate
  exceedances <- length(differences[which(differences < 0)])
  expected <- length(true_returns)*alpha
  ratio <- exceedances/expected
  final <- rbind(exceedances, expected, ratio)
  return(final)
}

lag_timeseries_mv <- function(time_series, max_lags) {
  
  lag_1 <- dplyr::lag(time_series)
  lag_df <- lag_1
  
  if (max_lags > 1) {
    for(i in 2:max_lags) {
      lag_df <- cbind(lag_df, dplyr::lag(time_series, i))
    }
  }
  
  names(lag_df) <- c(paste('lag_',c(1:max_lags), sep=''))
  return(lag_df)
}

prepare_returns <- function(df, scale=TRUE,
                            date_from, date_to,
                            log_corr=FALSE) {
  df$Date <- as.Date(df$Date)
  lagged_close <- dplyr::lag(df$Close)
  if (log_corr==TRUE) {
    log_returns <- log(df$Close/lagged_close)
  } else {
    log_returns <- (df$Close/lagged_close)
  }
  if (scale==TRUE) {
    log_returns <- scale(log_returns)
  }
  df$returns <- log_returns
  
  df <- df[which(df$Date > date_from & df$Date < date_to), ]
  
}

prepare_returns_commodities <- function(df, scale=TRUE,
                                        date_from, date_to,
                                        log_corr=FALSE) {
  # Correct Date
  df[,1] <- as.Date(df[,1])
  colnames(df)[1] <- c('Date')
  
  # Drop NA 
  df <- df[which(!is.na(df[,2])), ]
  
  # Calculate returns
  lagged_price <- dplyr::lag(df[,2])
  
  log_returns <- log(df[,2]/lagged_price)
  
  if (scale==TRUE) {
    log_returns <- scale(log_returns)
  }
  
  df <- cbind(df, log_returns)
  names(df)[3] <- paste('log_return_', names(df)[2], sep='')
  
  df <- df[which(df[,1] > date_from & df[,1] < date_to), ]
  df <- df[,c(1,2,3)]
  return(df)
}

quantile_loss <- function(q, y, f) {
  error <- y - f
  k_mean(k_maximum(q * error, (q - 1) * error), axis = 2)
}

lag_timeseries <- function(time_series, max_lags) {
  
  lag_1 <- dplyr::lag(time_series)
  lag_df <- cbind(time_series, lag_1)
  
  if (max_lags > 1) {
    for(i in 2:max_lags) {
      lag_df <- cbind(lag_df, dplyr::lag(time_series, i))
    }
  }
  
  names(lag_df) <- c('y',paste('lag_',c(1:max_lags), sep=''))
  return(lag_df)
}

exceeded_estimation <- function(VaR_estimate, true_returns, alpha) {
  differences <- true_returns - VaR_estimate
  exceedances <- length(differences[which(differences < 0)])
  expected <- length(true_returns)*alpha
  ratio <- exceedances/expected
  final <- rbind(exceedances, expected, ratio)
  return(final)
}

