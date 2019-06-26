##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Multivariate Time Series Simulation
##         COvariance models for MV-Normal and
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
library(forecast)

rm(list=ls())
gc()

columns_covariance <- matrix(c('iteration', 'MSPE_NAIVE', 'MSPE_MGARCH', 'MSPE_ANN', 'MSPE_SANN'), nrow=1, ncol=5)
write.table(columns_covariance, file = './code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/covariance_nonnormal.csv',
            sep=";",
            row.names = FALSE, col.names = FALSE)

#### Data generation ####
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
  error_matrix <- rmst(n_samples, xi, Omega, alpha, nu)
  
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
  
  # True Omegas 
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
  
  ## Fit MGARCH model
  #### MGARCH #### 
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
  
  covariance_forecast <- rcov(forecast_class)
  
  covariance_matrix_mgarch <- matrix(NA, nrow=out_sample, ncol=all_covars)
  for (i in 1:out_sample) {
    covariance_matrix_mgarch[i, ] <- unname(covariance_forecast[[i]][,,1])[upper.tri(unname(covariance_forecast[[i]][,,1]))]
  }
  
  mpse_mgarch_list <- list()
  for (i in 1:all_covars) {
    mspe <- sqrt(mean(((covariance_matrix_mgarch[2:100,i])-omega_covar[1001:1099,i])^2))
    mpse_mgarch_list[[i]] <- mspe
  }
  
  avg_mspe_mgarch <- mean(unlist(mpse_mgarch_list))
  
  ## SANN
  # Set up covariance data
  garch_mat <- returns_fin
  garch_mat_squared <- t(apply(garch_mat, 1, FUN=function(x) extract_upper_tri(x)))
  garch_lagged_1 <- dplyr::lag(garch_mat_squared, 1)
  garch_mat_target <- t(apply(garch_mat, 1, FUN=function(x) extract_upper_target(x)))

  linear_part_in_sample <- matrix(NA, nrow=(n-out_sample-1), ncol=all_covars)
  linear_part_out_sample <- matrix(NA, nrow=out_sample, ncol=all_covars)
  resid_part_in <- matrix(0, nrow=(n-out_sample-1), ncol=all_covars)
  resid_part_out <- matrix(0, nrow=(out_sample), ncol=all_covars)
  
  
  for (col in 1:all_covars) {
    garch_single <- garch_mat_target[,col]
    
    data_single <- garch_single
    lagged_single <- dplyr::lag(data_single)
    df_linear <- data.frame(cbind(data_single, lagged_single))
    names(df_linear) <- c('y', 'lag_y')
    lm_train <- lm(y ~ ., data=df_linear[1:(n-out_sample), ])
    pred_lin_in <- predict(lm_train)
    pred_lin_out <- predict(lm_train, newdata=df_linear[(n-out_sample+1):n, ])
    
    linear_part_in_sample[, col] <- pred_lin_in
    linear_part_out_sample[, col] <- pred_lin_out
    resid_part_in[, col] <-  pred_lin_in - omega_covar[2:999, col] 
    resid_part_out[, col] <- pred_lin_out - omega_covar[1000:1099, col]
  }
  
  ## Set up ANN-data
  train <- n-out_sample-1
  test <- train+1
  
  garch_mat_squared <- resid_part_in
  garch_lagged <- dplyr::lag(resid_part_in)
  lagged_out_resid <- dplyr::lag(resid_part_out)
  
  x_train <- array_reshape(garch_lagged[2:train, ], c(train-1, all_covars))
  y_train <- array_reshape(garch_mat_squared[2:train, ], c(train-1, all_covars))
  x_test <- array_reshape(lagged_out_resid[2:100, ], c(99, all_covars))
  
  use_session_with_seed(42)
  sann_part <- keras_model_sequential() %>%
    layer_dense(units = 150, activation = 'relu', input_shape = c(all_covars), activity_regularizer = regularizer_l1(l=0.000000002)) %>%
    layer_dense(units =20 , activation = 'relu', activity_regularizer = regularizer_l1(l=0.000000002)) %>%
    layer_dense(units=(all_covars), activation='linear',
                kernel_constraint = constraint_nonneg())
  
  sann_part %>% compile(
    optimizer = optimizer_adagrad(),
    loss = 'mse'
  )
  
  # Fit
  history <- sann_part %>%
    fit(x= x_train,
        y=y_train, steps_per_epoch = 200,
        epochs = 20, verbose = T)
  
  prediction_sann <- sann_part %>%
    predict(x_test)
  
  combined <- linear_part_out_sample[2:100, ] + prediction_sann
  
  mspe_sann <- list() 
  for (i in 1:all_covars) {
    mspe <- sqrt(mean((combined[,i] - omega_covar[1001:1099,i])^2))
    mspe_sann[[paste('mspe_',i, sep='')]] <- mspe
  }
  avg_mspe_sann <- mean(unlist(mspe_sann))
  
  ## Completely nonparametric ann
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
    layer_dense(units = 150, activation = 'relu', input_shape = c(num_var), activity_regularizer = regularizer_l1(l=0.000002)) %>%
    layer_dense(units =20 , activation = 'relu', activity_regularizer = regularizer_l1(l=0.000002)) %>%
    layer_dense(units=(num_var), activation='linear')
  
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
    naive <- mean(omega_covar[900:999, i])
    mspe <- sqrt(mean((naive - omega_covar[1001:1099,i])^2))
    mspe_naive[[paste('mspe_',i, sep='')]] <- mspe
  }
  avg_naive <- mean(unlist(mspe_naive))
  
  storeready_results <- matrix(c(j, avg_naive,avg_mspe_mgarch, avg_mspe_ann_pure, avg_mspe_sann), nrow=1, ncol=5)
  
  write.table(storeready_results, file = './code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/covariance_nonnormal.csv',
              sep=";", append = TRUE,
              row.names = FALSE, col.names = FALSE)
}