get_lagged_data <- function(lags, input_data) {
  x_tmp <- matrix(nrow = length(input_data), ncol=0)
  for (i in 1:lags) {
    lag_name <- paste("y", i,"lag", sep = "_")
    lag_vec <- dplyr::lag(input_data, i)
    x_tmp <- cbind(x_tmp, lag_vec)
  }
  
  x_tmp <- x_tmp[(lags+1):dim(x_tmp)[1], ]
  
  x_train <- array_reshape(x_tmp, c(dim(x_tmp)[1], 1*lags))
  y_train <- input_data[(lags+1):length(input_data)[1]]
  
  return(list(x_train, y_train))
}
