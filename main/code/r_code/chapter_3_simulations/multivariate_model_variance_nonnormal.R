##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Multivariate Time Series Simulation
##         Variance models for MV-Normal and
##         Skew-t distribution
##
## Tested: Windows >= 8
##----------------------------------------------##

rm(list=ls())
gc()

library(sn)
library(MASS)
library(rugarch)
library(rmgarch)
library(parallel)
library(keras)

distribution <- 'normal' # Chose 'normal' for baseline estimation

if(distribution == 'normal') {
  column_names_normal <- matrix(c('iteration', 'MSPE_NAIVE', 'MSPE_MGARCH', 'MSPE_ANN', 'MSPE_SANN'), nrow=1, ncol=5)
  write.table(column_names_normal, file = './code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/variance_normal.csv',
              sep=";",
              row.names = FALSE, col.names = FALSE)
  
} else {
  column_names_nonnormal <- matrix(c('iteration', 'MSPE_NAIVE', 'MSPE_MGARCH', 'MSPE_ANN', 'MSPE_SANN'), nrow=1, ncol=5)
  write.table(column_names_nonnormal, file = './code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/variance_nonnormal.csv',
              sep=";",
              row.names = FALSE, col.names = FALSE)
}

#### Data generation ####
# The data generation is unstable for some initializations
# we will skip the seed where this happens and simulate with the first 
# 50 seeds that do not cause instability
sims <- c(1:62)
sims_not <- rep(TRUE, 62)
fails <- c(11,17,20,23,29,34,55)
sims_not[fails] <- FALSE
sims_cleaned <- sims[sims_not]

for (j in sims_cleaned[1:50]) {
  ## Simulation of Data
  num_var = 10
  skew_param = -1.5
  nu = 10
  burn_in = 100
  n = 1000
  n_samples = (burn_in + n)
  out_sample = 100
  
  
  # Helpers
  source('./code/r_code/helper_functions/data_generation.R')
  
  set.seed(j)
  xi <- runif(num_var)
  Omega <- diag(num_var)
  correlation_mat <- positive_matrix(num_var, ev=runif(num_var,-1,1))
  Omega[which(row(Omega)!=col(Omega))] <- correlation_mat[which(row(correlation_mat)!=col(correlation_mat))]
  alpha <- c(rep(skew_param,num_var))
  
  if(distribution == 'normal') {
    error_matrix <- mvrnorm(n=n_samples, mu=rep(0,num_var), Sigma=Omega)
  } else {
    error_matrix <- rmst(n_samples, xi, Omega, alpha, nu)
  }
  err_std <- scale(error_matrix)
  
  Omega_list <- list()
  Omega_list[[1]] <- round(Omega, 5)
  all_covars <- sum(upper.tri(Omega_list[[1]], diag = F))
  
  return_t <- matrix(nrow = 1100, ncol=num_var)
  sigma_mat <- matrix(nrow= 1100, ncol= num_var)
  rho_t <- matrix(nrow= 1100, ncol= all_covars)
  
  for(i in 1:1099){
    
    return_t[i, ] <- matrix(expm::sqrtm(Omega_list[[i]]) %*% err_std[i,], nrow=1, ncol=num_var)
    sigma_mat[(i+1), ] = 0.3 + 0.55*return_t[i, ]^2
    H_t <- round(diag(sqrt(sigma_mat[i+1, ])), 5)
    rho_t[i, ] <- 0.5 + 0.2*cos((2*pi*(i))/1000)
    
    R_t <- diag(nrow=num_var)
    R_t[upper.tri(R_t)] <- rho_t[i, ]
    R_f <- R_t + t(R_t)
    diag(R_f) <- 1
    R_t <- round(R_f, 5)
    
    Omega_list[[i+1]] <- H_t %*% R_t %*% H_t
  }
  
  # True Omegas (corresponds to R_t in the thesis)
  omega_var <- matrix(NA, ncol=num_var, nrow = 1100)
  omega_covar <- matrix(NA, ncol=all_covars, nrow=1100)
  
  for (i in 1:1099){
    omega_var[i,] <- diag(Omega_list[[i]])
    omega_covar[i, ] <- Omega_list[[i]][upper.tri(Omega_list[[1]], diag = F)]
  }
  
  returns_fin <- return_t[complete.cases(return_t), ]
  n <- nrow(returns_fin)
  out_sample <- 100
  garch_data <- returns_fin
  
  ## MGARCH
  num_cores <- detectCores()

  mgarch_specification = ugarchspec(mean.model = list(armaOrder = c(1, 1)),
                                    variance.model = list(garchOrder = c(1,1),
                                                          model = 'sGARCH'),
                                    distribution.model = 'norm')
  
  # Replicate for all columns
  all_specs = multispec(replicate(num_var,
                                  mgarch_specification))
  
  # Specify MGARCH model (DCC)
  specgarch = dccspec(uspec = all_specs,
                      dccOrder = c(1, 1),
                      distribution = 'mvnorm')
  
  # Run on local cluster
  cl = makePSOCKcluster(num_cores)
  multf = multifit(all_specs, garch_data[1:(n-out_sample),], cluster=cl)
  fit_mgarch = dccfit(specgarch, data = garch_data, fit.control = list(eval.se = TRUE), fit = multf,
                      out.sample = out_sample, cluster=cl)
  stopCluster(cl)
  
  # Create forecasting class
  forecast_class <- dccforecast(fit = fit_mgarch,
                                n.ahead = 1,
                                n.roll=(out_sample-1))
  
  in_variance <- sigma(fit_mgarch)^2
  out_variance <- sigma(forecast_class)^2
  
  var_norm_list <- list()
  for(i in 1:num_var) {
    mspe <- sqrt(mean((t(out_variance[1,,])[2:100, i] - omega_var[1001:1099,i])^2))
    var_norm_list[[i]] <- mspe
  }
  mspe_mgarch_pure <- mean(unlist(var_norm_list))
  
  ## SANN-residuals
  linear_part_in_sample <- matrix(NA, nrow=(n-out_sample-1), ncol=num_var)
  linear_part_out_sample <- matrix(NA, nrow=out_sample, ncol=num_var)
  resid_part_in <- matrix(0, nrow=(n-out_sample-1), ncol=num_var)
  resid_part_out <- matrix(0, nrow=(out_sample), ncol=num_var)
  
  for (col in 1:num_var) {
    garch_single <- garch_data[,col]
    data_single <- garch_single^2
    lagged_single <- dplyr::lag(data_single)
    df_linear <- data.frame(cbind(data_single, lagged_single))
    names(df_linear) <- c('y', 'lag_y')
    lm_train <- lm(y ~ ., data=df_linear[1:(n-out_sample), ])
    pred_lin_in <- predict(lm_train)
    pred_lin_out <- predict(lm_train, newdata=df_linear[(n-out_sample+1):n, ])
    
    linear_part_in_sample[, col] <- pred_lin_in
    linear_part_out_sample[, col] <- pred_lin_out
    resid_part_in[, col] <-  omega_var[2:999, col]- pred_lin_in
    resid_part_out[, col] <- omega_var[1000:1099, col]- pred_lin_out
  }
  
  ## Set up ANN-data
  train <- n-out_sample-1
  test <- train+1
  
  garch_mat_squared <- resid_part_in
  garch_lagged <- dplyr::lag(resid_part_in)
  lagged_out_resid <- dplyr::lag(resid_part_out)
  
  x_train <- array_reshape(garch_lagged[2:train, ], c(train-1, num_var))
  y_train <- array_reshape(garch_mat_squared[2:train, ], c(train-1, num_var))
  
  x_test <- array_reshape(lagged_out_resid[2:100, ], c(99, num_var))

  use_session_with_seed(42)
  sann_part <- keras_model_sequential() %>%
    layer_dense(units = 150, activation = 'relu', input_shape = c(num_var), activity_regularizer = regularizer_l1(l=0.002)) %>%
    layer_dense(units =20 , activation = 'relu', activity_regularizer = regularizer_l1(l=0.002)) %>%
    layer_dense(units=(num_var), activation='linear', activity_regularizer = regularizer_l1(l=0.000015),
                kernel_constraint = constraint_nonneg())
  
  sann_part %>% compile(
    optimizer = optimizer_adagrad(),
    loss = 'mse'
  )
  
  # Fit
  history <- sann_part %>%
    fit(x= x_train,
        y=y_train, steps_per_epoch = 200,
        epochs = 50, verbose = F)
  
  prediction_sann <- sann_part %>%
    predict(x_test)
  combined <- linear_part_out_sample[2:100, ] + prediction_sann
 
  mspe_sann <- list() 
  for (i in 1:num_var) {
    mspe <- sqrt(mean((combined[,i] - omega_var[1001:1099,i])^2))
    mspe_sann[[paste('mspe_',i, sep='')]] <- mspe
  }
  avg_mspe_sann <- mean(unlist(mspe_sann))
  
  ## Completely nonparametric ANN
  garch_mat_squared <- garch_data^2
  garch_lagged <- dplyr::lag(garch_mat_squared)

  train <- (n_samples-out_sample-1)
  test <- train + 1
  
  keras_mat_train <- garch_lagged[2:train, ]
  keras_mat_test <- garch_lagged[test:1099, ]
  
  y_mat_train <- garch_mat_squared[2:train,]
  y_mat_test <- garch_mat_squared[test:1099,]
  
  x_train <- array_reshape(as.matrix(keras_mat_train), c(train-1, num_var))
  y_train <- array_reshape(as.matrix(y_mat_train), c(train-1, num_var))
  
  x_test <- array_reshape(as.matrix(keras_mat_test), c(100, num_var))
  y_test <- array_reshape(as.matrix(y_mat_test), c(100, num_var))
  
  # Run ANN
  use_session_with_seed(42)
  ann_pure <- keras_model_sequential() %>%
    layer_dense(units = 50, activation = 'relu', input_shape = c(num_var), activity_regularizer = regularizer_l1(l=0.002)) %>%
    layer_dense(units =20 , activation = 'relu', activity_regularizer = regularizer_l1(l=0.002)) %>%
    layer_dense(units=(num_var), activation='linear', activity_regularizer = regularizer_l1(l=0.000015))
  
  ann_pure %>% compile(
    optimizer = optimizer_adagrad(),
    loss = 'mse'
  )
  ann_pure %>%
    fit(x= x_train,
        y=y_train, steps_per_epoch = 200,
        epochs = 50, verbose = F)
  
  prediction_ann <- ann_pure %>%
    predict(x_test)
  
  mspe_ann_pure <- list()
  for (i in 1:num_var) {
    mspe <- sqrt(mean((prediction_ann[2:100] - omega_var[1001:1099,i])^2))
    mspe_ann_pure[[paste('mspe_',i, sep='')]] <- mspe
  }
  avg_mspe_ann_pure <- mean(unlist(mspe_ann_pure))
  
  mspe_naive <- list()
  for (i in 1:num_var) {
    naive <- mean(omega_var[900:999, i])
    mspe <- sqrt(mean((naive - omega_var[1001:1099,i])^2))
    mspe_naive[[paste('mspe_',i, sep='')]] <- mspe
  }
  avg_naive <- mean(unlist(mspe_naive))
  
  storeready_results <- matrix(c(j, avg_naive,mspe_mgarch_pure, avg_mspe_ann_pure, avg_mspe_sann), nrow=1, ncol=5)
  
  if(distribution == 'normal') {
    write.table(storeready_results, file = './code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/variance_normal.csv',
                sep=";", append = T,
                row.names = FALSE, col.names = FALSE)
    
  } else {
    write.table(storeready_results, file = './code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/variance_nonnormal.csv',
                sep=";", append = T,
                row.names = FALSE, col.names = FALSE)
  }
  
  if(j==1 & distribution!='normal') {
    write.table(omega_var[1000:1099, ], file = './code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/truth_nonnormal.csv',
                sep=";", 
                row.names = FALSE, col.names = FALSE)
    write.table(combined, file = './code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/SANN_prediction_nonnormal.csv',
                sep=";", 
                row.names = FALSE, col.names = FALSE)
    write.table(t(out_variance[1,,])[2:100, ], file = './code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/MGARCH_prediction_nonnormal.csv',
                sep=";", 
                row.names = FALSE, col.names = FALSE)
  }
}