generate_data_iid <- function(size, mean, sd, x_from, x_to, break_at) {
  x <- seq(from=x_from, to=x_to, by=(x_to-x_from)/size)
  noise <- rnorm(length(x), mean=mean, sd=sd)
  y_true <- 0.4*(x-10)^3 + 0.1*(x/7)^7 + sin(x*2)*600 + ifelse(test = (x>break_at), yes = -sin(x*2)*600 - 200 -sin(x*2)*200, no = 0)
  y_noise <- y_true + noise
  
  return(list(x, y_noise, y_true))
}

generate_data_iid_seed <- function(size, mean, sd, x_from, x_to, break_at) {
  x <- seq(from=x_from, to=x_to, by=(x_to-x_from)/size)
  y_true <- 0.4*(x-10)^3 + 0.1*(x/7)^7 + sin(x*2)*600 + ifelse(test = (x>break_at), yes = -sin(x*2)*600 - 200 -sin(x*2)*200, no = 0)
  y_noise <- y_true 
  
  return(list(x, y_noise, y_true))
}

generate_high_dim_add_iid <- function(size, mean, sd,
                                      no_relevant, no_irrelevant,
                                      noise_x_from, noise_x_to,
                                      x_from, x_to,
                                      function_list) {
  # Draw X
  x_noise <- as.matrix(replicate(no_irrelevant, runif(size, noise_x_from, noise_x_to)))
  x_true <- as.matrix(replicate(no_relevant, runif(size, x_from, x_to)))
  x_matrix <- cbind(x_true, x_noise)
  
  # Draw error
  err <- rnorm(size, mean = mean, sd=sd)
  
  # Simulate Y according to function input, not really a nice way - but hey - it works
  y_true <- 0
  for (i in 1:no_relevant) {
    if (i < 11) {
      y_true <- y_true + function_list[[i]](x_matrix[,i])
    }
    else {
      y_true <- y_true + function_list[[11]](x_matrix[,i], i)
    }
  }
  y <- y_true + err
  
  # Set up dataframe (for use with np-package, as a bug prevents predictions with matrices as input)
  df <- data.frame(cbind(y,x_matrix))
  names(df) <- c('y', paste("v_",c(1:(no_relevant + no_irrelevant)), sep = ""))
  
  return(list(df, y_true))
}

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

positive_matrix <- function (n, ev = runif(n, 0, 10)) 
{
  Z <- matrix(ncol=n, rnorm(n^2))
  decomp <- qr(Z)
  Q <- qr.Q(decomp) 
  R <- qr.R(decomp)
  d <- diag(R)
  ph <- d / abs(d)
  O <- Q %*% diag(ph)
  Z <- t(O) %*% diag(ev) %*% O
  return(Z)
}

extract_upper_tri <- function(asset_row) {
  varcov_mat <- asset_row %*% t(asset_row)
  lower_tri_mat <- varcov_mat[upper.tri(varcov_mat, diag = T)]
  return(lower_tri_mat)
}

extract_upper_target <- function(asset_row) {
  varcov_mat <- asset_row %*% t(asset_row)
  lower_tri_mat <- varcov_mat[upper.tri(varcov_mat, diag = F)]
  return(lower_tri_mat)
}