##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Univariate CAViaR model
## Tested: Apple-Darwin
##----------------------------------------------##

rm(list=ls())
gc()

library(dplyr)
library(rugarch)
library(sfsmisc)
library(keras)
library(quantreg)
library(stargazer)

caviar_file <- './code/r_code/results_application/univariate_models/coefs_univariate_caviar'

if(file.exists(caviar_file)) {
  load(file='./code/r_code/results_application/univariate_models/var_forecast_sann')
  load(file='./code/r_code/results_application/univariate_models/var_forecast_sann_rec')
  load(file='./code/r_code/results_application/univariate_models/var_forecast_garch')
  
  load(file='./code/r_code/results_application/univariate_models/var_in_sann_rec')
  load(file='./code/r_code/results_application/univariate_models/var_in_sann')
  load(file='./code/r_code/results_application/univariate_models/var_in_garch')
  load(file= './code/r_code/results_application/univariate_models/coefs_univariate_caviar')
  
  load(file='./code/r_code/results_application/univariate_models/prepred_ge_data')
} else {
  source('./code/r_code/helper_functions/result_transformation.R')
  # Read and prepare data
  data_ge <- read.csv(file = './data/stocks/ge.us.txt')
  prepared_ge <- prepare_returns(data_ge, scale = TRUE,
                                 date_from = '2000-01-01', date_to = '2012-12-31',
                                 log_corr = TRUE)
  
  # Change for use with keras models
  variance <- prepared_ge$returns^2
  df_simple <- lag_timeseries(variance, max_lags = 3)
  
  hold_out <- 1000
  train_to <- dim(df_simple)[1]-hold_out
  prepared_keras <- get_keras_data(input_data = df_simple, train = train_to,
                                   position_linear = c(2:4), position_nonlinear = c(2:4),
                                   position_y = c(1), skip_first = 3)
  
  use_session_with_seed(42)
  nonlinear_input <- layer_input(shape = c(3), name='nonlin_input')
  nonlinear <- nonlinear_input %>%
    layer_dense(units =80, activation = 'relu' , activity_regularizer = regularizer_l1(l=0.00001)) %>%
    layer_dense(units = 5 , activation = 'relu')
  
  # Add linear layer and combine
  linear_input <- layer_input(shape = c(3), name = 'linear_input')
  both_layers <- layer_concatenate(list(nonlinear, linear_input))
  output <- both_layers %>%
    layer_dense(units=1, activation = 'linear', use_bias = T, kernel_constraint = constraint_nonneg())
  model_hybrid <- keras_model(inputs = list(nonlinear_input, linear_input),
                              outputs = output)
  quantile <- 0.99
  # Add estimation
  model_hybrid %>% compile(
    optimizer = optimizer_adagrad(),
    loss =  function(y_true, y_pred)
      quantile_loss(quantile, y_true, y_pred)
  )
  
  # Fit Hybrid
  model_hybrid %>%
    fit(x= list(nonlin_input=prepared_keras[[1]][[3]], linear_input=prepared_keras[[1]][[2]]),
        y=prepared_keras[[1]][[1]], steps_per_epoch = 200,
        epochs = 50, verbose = T)
  
  # In and out of sample predictions
  prediction_hybrid <- model_hybrid %>%
    predict(list(nonlin_input=prepared_keras[[2]][[3]], linear_input=prepared_keras[[2]][[2]]))
  in_sample_hybrid <- model_hybrid %>%
    predict(list(nonlin_input=prepared_keras[[1]][[3]], linear_input=prepared_keras[[1]][[2]]))
  
  ## GARCH-Model
  out_sample <- 1000
  
  univariate_garch <- ugarchspec(mean.model=list(armaOrder=c(1,1)),
                                 variance.model=list(garchOrder=c(2,1)),
                                 distribution.model = "norm")
  
  univariate_fit <- ugarchfit(data=prepared_ge$returns,
                              spec=univariate_garch,
                              out.sample = out_sample,
                              solver = 'hybrid')
  
  forecast_univariate <- ugarchforecast(univariate_fit,
                                        data = NULL,
                                        n.ahead = 1,
                                        n.roll = (out_sample-1),
                                        out.sample = (out_sample))
  
  ## Recurrent SANN
  
  initial_bvalue <- rep(unname(quantile(prepared_keras[[1]][[1]], probs=c(0.99))), dim(prepared_keras[[1]][[2]])[1])
  hold_out <- 1000
  num_it <- 5
  
  for (i in 1:num_it) {
    
    df_simple <- lag_timeseries(variance, max_lags = 3)
    train_to <- dim(df_simple)[1]-hold_out
    prepared_keras <- get_keras_data(input_data = df_simple, train = train_to,
                                     position_linear = c(2:3), position_nonlinear = c(2:3),
                                     position_y = c(1), skip_first = 3)
    if (i == 1) {
      prepared_keras[[1]][[2]] <- cbind(prepared_keras[[1]][[2]], initial_bvalue)
    } else {
      prepared_keras[[1]][[2]] <- cbind(prepared_keras[[1]][[2]], y_hat_lagged)
    }
    
    use_session_with_seed(42)
    nonlinear_input <- layer_input(shape = c(2), name='nonlin_input')
    nonlinear <- nonlinear_input %>%
      layer_dense(units =80, activation = 'relu', activity_regularizer = regularizer_l1(l=0.0000001)) %>%
      layer_dense(units = 5 , activation = 'relu')
    
    # Add linear layer and combine
    linear_input <- layer_input(shape = c(3), name = 'linear_input')
    both_layers <- layer_concatenate(list(nonlinear, linear_input))
    output <- both_layers %>%
      layer_dense(units=1, activation = 'linear', use_bias = T,
                  kernel_constraint = constraint_nonneg())
    model_hybrid_rec <- keras_model(inputs = list(nonlinear_input, linear_input),
                                    outputs = output)
    quantile <- 0.99
    # Add estimation
    model_hybrid_rec %>% compile(
      optimizer = optimizer_adagrad(), 
      loss =  function(y_true, y_pred)
        quantile_loss(quantile, y_true, y_pred)
    )
    
    # Fit Hybrid
    model_hybrid_rec %>%
      fit(x= list(nonlin_input=prepared_keras[[1]][[3]], linear_input=prepared_keras[[1]][[2]]),
          y=prepared_keras[[1]][[1]], steps_per_epoch = 200,
          epochs = 50, verbose = F)
    
    y_hat <- model_hybrid_rec %>%
      predict(list(nonlin_input=prepared_keras[[1]][[3]], linear_input=prepared_keras[[1]][[2]]))
    
    y_hat_lagged <- matrix(c(dplyr::lag(y_hat)), ncol=1)
    y_hat_lagged[1,1] <- y_hat_lagged[2,1]
  }
  
  # Forecast with recurrence relation
  init_test <- initial_bvalue <- rep(unname(quantile(prepared_keras[[1]][[1]][(nrow(prepared_keras[[1]][[1]])-100):nrow(prepared_keras[[1]][[1]])],
                                                     probs=c(0.99))), 1000)
  for (i in 1:50) {
    if (i == 1) {
      x_lin_unpreped <- cbind(prepared_keras[[2]][[2]], init_test)
    } else {
      x_lin_unpreped <- cbind(prepared_keras[[2]][[2]], lagged_pred)
    }
    
    prediction_hybrid_caviar <- model_hybrid_rec %>%
      predict(list(nonlin_input=prepared_keras[[2]][[3]], linear_input=x_lin_unpreped))
    # Avoid numerical issues
    prediction_hybrid_caviar[which(prediction_hybrid_caviar == 0 | prediction_hybrid_caviar < 0)] <- 0.00001
    
    prediction_hybrid_prep <- (prediction_hybrid_caviar)
    
    lagged_pred <- dplyr::lag(prediction_hybrid_prep)
    lagged_pred[1,] <- lagged_pred[2,]
  }
  
  ## Isolate Quantile parameter
  layer_name='concatenate_1'
  last_layer <- keras_model(inputs = model_hybrid_rec$input,
                            outputs = get_layer(model_hybrid_rec, layer_name)$output)
  
  df_layer <- as.data.frame(predict(last_layer, list(prepared_keras[[1]][[3]], prepared_keras[[1]][[2]])))
  df_layer$y <- prepared_keras[[1]][[1]]
  
  summary(lm(y ~., data=df_layer))
  quantile_reg <- rq(y ~ ., data=df_layer[,c(1:3, 6:9)], tau = 0.99)
  
  coefs_quant <- c(unname(coef(summary(quantile_reg))[,'Value'])[7], unname(coef(summary(quantile_reg))[,'Std. Error'])[7])
  
  ## Store results
  var_forecast_hybrid_rec <- -sqrt(prediction_hybrid_caviar)
  var_forecast_hybrid <- -sqrt(prediction_hybrid)
  var_forecast_garch <- as.numeric(quantile(forecast_univariate, probs = 0.01))
  
  var_in_hybrid_rec <- -sqrt(y_hat)
  var_in_hybrid <- -sqrt(in_sample_hybrid)
  var_in_garch <- as.numeric(quantile(univariate_fit, probs = 0.01))
  
  save(var_forecast_hybrid, file='./code/r_code/results_application/univariate_models/var_forecast_sann')
  save(var_forecast_hybrid_rec, file='./code/r_code/results_application/univariate_models/var_forecast_sann_rec')
  save(var_forecast_garch, file='./code/r_code/results_application/univariate_models/var_forecast_garch')
  
  save(var_in_hybrid_rec, file='./code/r_code/results_application/univariate_models/var_in_sann_rec')
  save(var_in_hybrid, file='./code/r_code/results_application/univariate_models/var_in_sann')
  save(var_in_garch, file='./code/r_code/results_application/univariate_models/var_in_garch')
  save(coefs_quant, file = './code/r_code/results_application/univariate_models/coefs_univariate_caviar')
  save(prepared_ge, file='./code/r_code/results_application/univariate_models/prepred_ge_data' )
}

#### Diagnostics ####

GARCH_number_p_01 <- VaRTest(prepared_ge$returns[(nrow(prepared_ge)-999):nrow(prepared_ge)], var_forecast_garch, alpha = 0.01)
GARCH_duration_p_01 <- VaRDurTest(prepared_ge$returns[(nrow(prepared_ge)-999):nrow(prepared_ge)], var_forecast_garch, alpha = 0.01)

SANN_number_p_01 <- VaRTest(prepared_ge$returns[(nrow(prepared_ge)-999):nrow(prepared_ge)], var_forecast_hybrid, alpha = 0.01)
SANN_duration_p_01 <- VaRDurTest(prepared_ge$returns[(nrow(prepared_ge)-999):nrow(prepared_ge)], var_forecast_hybrid, alpha = 0.01)

RSANN_number_p_01 <- VaRTest(prepared_ge$returns[(nrow(prepared_ge)-999):nrow(prepared_ge)], var_forecast_hybrid_rec, alpha = 0.01)
RSANN_duration_p_01 <- VaRDurTest(prepared_ge$returns[(nrow(prepared_ge)-999):nrow(prepared_ge)], var_forecast_hybrid_rec, alpha = 0.01)

exceedances_garch <- GARCH_number_p_01$actual.exceed
exceedances_sann <- SANN_number_p_01$actual.exceed
exceedances_rsann <- RSANN_number_p_01$actual.exceed

pval_exceedances_garch <- GARCH_number_p_01$uc.LRp
pval_exceedances_sann <- SANN_number_p_01$uc.LRp
pval_exceedances_rsann <- RSANN_number_p_01$uc.LRp

pval_duration_garch <- GARCH_duration_p_01$LRp
pval_duration_sann <- SANN_duration_p_01$LRp
pval_duration_rsann <- RSANN_duration_p_01$LRp

uncond_quantile <- rep(-unname(quantile(prepared_ge$returns[(nrow(prepared_ge)-999):nrow(prepared_ge)], probs=0.01)),1000) 
int_var_garch <- integrate.xy(x = seq(1,1000,1), c(-var_forecast_garch))
int_var_sann <- integrate.xy(x = seq(1,1000,1), c(-var_forecast_hybrid))
int_var_sann_rec <- integrate.xy(x = seq(1,1000,1), c(-var_forecast_hybrid_rec))
int_var_uncond <- integrate.xy(x = seq(1,1000,1), uncond_quantile)

improve_garch <- ((int_var_garch - int_var_uncond)/int_var_uncond)*100
improve_sann <- ((int_var_sann - int_var_uncond)/int_var_uncond)*100
improve_rsann <- ((int_var_sann_rec - int_var_uncond)/int_var_uncond)*100

exceed_vec <- c(exceedances_garch, exceedances_sann, exceedances_rsann)
pval_exced_vec <- c(pval_exceedances_garch, pval_exceedances_sann, pval_exceedances_rsann)
pval_dur_vec <- c(pval_duration_garch, pval_duration_sann, pval_duration_rsann)
improve_vec <- c(improve_garch, improve_sann, improve_rsann)
param_vec <- c(NA, NA, coefs_quant[1])
sd_vec <- c(NA, NA, coefs_quant[2])

results_univariate <- rbind(exceed_vec, pval_exced_vec, pval_dur_vec, improve_vec, param_vec, sd_vec)
colnames(results_univariate) <- c('GARCH(1,1)', 'SANN(2,0)', 'SANN(2,1)')
rownames(results_univariate) <- c('Exceedances', 'Number', 'Duration', 'Improvement', 'Parameter', 'Std. dev')

table_univariate <- './code/r_code/results_application/univariate_models/table_univariate_results'
if(file.exists(table_univariate)) {
 load(file='./code/r_code/results_application/univariate_models/table_univariate_results')
} else {
  save(results_univariate, file='./code/r_code/results_application/univariate_models/table_univariate_results')
}

