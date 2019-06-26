##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Multivariate Time Series Simulation
##         Variance estimates for nonlinear
##         variance process models
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

column_names_nonlinear_multivar <- matrix(c('iteration', 'MSPE_NAIVE', 'MSPE_MGARCH', 'MSPE_ANN', 'MSPE_SANN'), nrow=1, ncol=5)
write.table(column_names_nonlinear_multivar, file = './code/r_code/results_simulation/results_multivariate_timeseries/form_misspecified/variance_nonlinear.csv',
            sep=";",
            row.names = FALSE, col.names = FALSE)

for (j in 1:50) {
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
  sig <- diag(num_var)
  correlation_mat <- positive_matrix(num_var, ev=runif(num_var,0,1.5))
  error_matrix <- mvrnorm(n = n_samples, mu=rep(0,num_var), correlation_mat)  
  err_std <- scale(error_matrix)

  # Initialize process matrix and run process
  alpha=0.7
  beta=-0.2
  alpha_0=1-alpha-beta
  
  garch_mat <- matrix(rep(0, n_samples*num_var), nrow = n_samples, ncol = num_var)
  sigma_mat <- err_std^2
  
  true_error <- matrix(rnorm(n_samples*num_var, mean = 0, sd=0.1), ncol=num_var, nrow=n_samples)

  for(i in 2:(n_samples-1)) {
    err_std[i, ] = sqrt(sigma_mat[i,])*err_std[i,]
    garch_mat[i, ] = err_std[i, ]
    sigma_mat[(i+1), ] = alpha_0 + ((alpha*err_std[i, ]^2 - beta*err_std[i-1, ]^2)/(1+err_std[i, ]^2 + err_std[i+1, ]^2)) +
                        true_error[i+1,] 
  }
  
  garch_mat <- scale(garch_mat+true_error)
  garch_data <- garch_mat[(burn_in + 1):n_samples, ]
  sigma_data <- sigma_mat[(burn_in + 1):n_samples, ]
  
  #### MGARCH model ####
  
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
  forecast_variance <- sigma(forecast_class)^2
  
  # Run diagnostics
  mspe_mgarch <- list()
  for (i in 1:num_var) {
    bias_corr_mgarch <- mean(sigma_data[801:900,i])-mean(forecast_variance[,i,])
    mspe <- sqrt(mean((sigma_data[(n-out_sample+1):n, i] - forecast_variance[,i,])^2))
    mspe_mgarch[[paste('mspe_',i, sep='')]] <- mspe
  }
  
  avg_mspe_mgarch <- mean(unlist(mspe_mgarch))
  
  ## ANNs
  # Pure first
  garch_mat_squared <- garch_mat^2
  garch_lagged <- dplyr::lag(garch_mat_squared)
  
  train <- (n_samples-out_sample)
  test <- train + 1
  
  keras_mat_train <- garch_lagged[(burn_in+1):train, ]
  keras_mat_test <- garch_lagged[test:n_samples, ]
  
  y_mat_train <- garch_mat_squared[(burn_in+1):train, ]
  y_mat_test <- garch_mat_squared[test:n_samples, ]
  
  x_train <- array_reshape(as.matrix(keras_mat_train), c(train-burn_in, num_var))
  y_train <- array_reshape(as.matrix(y_mat_train), c(train-burn_in, num_var))
  
  x_test <- array_reshape(as.matrix(keras_mat_test), c(n_samples-test+1, num_var))
  y_test <- array_reshape(as.matrix(y_mat_test), c(n_samples-test+1, num_var))
  
  #### Pure ANN ####
  use_session_with_seed(42)
  ann_pure <- keras_model_sequential() %>%
    layer_dense(units = 150, activation = 'relu', input_shape = c(10), activity_regularizer = regularizer_l1(l=0.0002)) %>%
    layer_dense(units =20 , activation = 'relu', activity_regularizer = regularizer_l1(l=0.0002)) %>%
    layer_dense(units=10, activation='linear', activity_regularizer = regularizer_l1(l=0.00025), kernel_constraint = constraint_nonneg())
  
  ann_pure %>% compile(
    optimizer = optimizer_adagrad(),
    loss = 'mse'
  )
  
  history <- ann_pure %>%
    fit(x= x_train,
        y=y_train, steps_per_epoch = 100,
        epochs = 50, verbose = F)
  
  prediction_ann <- ann_pure %>%
    predict(x_test)
  
  mspe_ann_pure <- list()
  for (i in 1:num_var) {
    mean_correction_ann <- mean(sigma_data[800:900,i ]) - mean(prediction_ann[,i])
    mspe <- sqrt(mean((sigma_data[(n-out_sample+1):n, i] - (prediction_ann[,i]+mean_correction_ann))^2))
    mspe_ann_pure[[paste('mspe_',i, sep='')]] <- mspe
  }
  
  avg_mspe_ann_pure <- mean(unlist(mspe_ann_pure))
  
  #### Semiparametric ANN ####
  use_session_with_seed(42)
  nonlinear_input <- layer_input(shape = c(10), name='nonlin_input')
  nonlinear <- nonlinear_input %>%
    layer_dense(units = 150, activation = 'relu', activity_regularizer = regularizer_l1(l=0.00015)) %>%
    layer_dense(units =20 , activation = 'relu', activity_regularizer = regularizer_l1(l=0.0015))
  
  # Add linear layer and combine
  linear_input <- layer_input(shape = c(10), name = 'linear_input')
  both_layers <- layer_concatenate(list(nonlinear, linear_input))
  output <- both_layers %>%
    layer_dense(units=10, activation = 'linear',
                activity_regularizer = regularizer_l1(l=0.00025), kernel_constraint = constraint_nonneg())
  model_hybrid <- keras_model(inputs = list(nonlinear_input, linear_input),
                              outputs = output)
  
  # Add estimation
  model_hybrid %>% compile(
    optimizer = optimizer_adagrad(),
    loss = 'mse'
  )
  
  model_hybrid %>%
    fit(x= list(nonlin_input=x_train, linear_input=x_train),
        y=y_train, steps_per_epoch = 100,
        epochs = 50, verbose = F)
  
  prediction_hybrid <- model_hybrid %>%
    predict(list(x_test, x_test))
  
  mspe_semi_ann <- list()
  for (i in 1:num_var) {
    mean_correction_semi <- mean(sigma_data[800:900,i ]) - mean(prediction_hybrid[,i])
    mspe <- sqrt(mean((sigma_data[(n-out_sample+1):n, i] - (prediction_hybrid[,i]+mean_correction_semi))^2))
    mspe_semi_ann[[paste('mspe_',i, sep='')]] <- mspe
  }
  
  avg_mspe_semi_ann <- mean(unlist(mspe_semi_ann))
  
  # Set naive Benchmark
  mspe_naive_list <- list()
  for (i in 1:num_var){
    mspe_naive <- sqrt(mean((sigma_data[901:1000,i] - mean(garch_mat_squared[800:900,i]))^2))
    mspe_naive_list[[i]] <- mspe_naive
  }
  
  mspe_naive <- mean(unlist(mspe_naive_list))
  
  results_mspe <- matrix(c(j, mspe_naive, avg_mspe_mgarch,
                           avg_mspe_semi_ann, avg_mspe_ann_pure), nrow=1, ncol=5)
  
  write.table(results_mspe, file = './code/r_code/results_simulation/results_multivariate_timeseries/form_misspecified/variance_nonlinear.csv',
              sep=";", append = TRUE, 
              row.names = FALSE, col.names = FALSE)
  
  rm(list=setdiff(ls(), c("simulation_runs", "sims_cleaned")))
  gc()
}

