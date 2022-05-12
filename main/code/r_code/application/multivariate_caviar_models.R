##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Multivariate CAViaR (MGARCH) models
## Tested: Windows >=8
##----------------------------------------------##
rm(list=ls())
gc()

library(tidyverse)
library(keras)
library(rmgarch)

source('./code/r_code/helper_functions/data_generation.R')
source('./code/r_code/helper_functions/result_transformation.R')

#### Data Prep ####
file_portfolio <- './code/r_code/results_application/multivariate_models/prepared_data/dataframe_portfolio'

if(file.exists(file_portfolio)) {
  load(file_portfolio)
} else {
  # Stocks
  data_ge <- read.csv(file = './data/stocks/ge.us.txt')
  data_walmart <- read.csv(file = '~/Desktop/data_thesis/stock_market/Data/Stocks/wmt.us.txt')
  data_mo <- read.csv(file = '~/Desktop/data_thesis/stock_market/Data/Stocks/mo.us.txt')
  
  # Commodities and exchange
  data_brent <- read.csv(file = './data/commodities/brent_crude.csv',
                         na.strings = '.')
  data_gold <- read.csv(file = './data/commodities/gold_price.csv',
                        na.strings = '.')
  data_chf <- read.csv(file = './data/commodities/usd_chf_exchange.csv', 
                       na.strings = '.')
  
  
  stock_list <- list(data_ge, data_walmart, data_mo)
  response_stock <- lapply(stock_list, function(x) prepare_returns(x, scale = TRUE,
                                                                   date_from = '2000-01-01', date_to = '2012-12-31'))
  
  commodities_list <- list(data_brent, data_gold, data_chf)
  
  response_com <- lapply(commodities_list,
                         function(x) prepare_returns_commodities(x, scale = TRUE,
                                                                 date_from = '2000-01-01',
                                                                 date_to = '2012-12-31'))
  combined <- c(response_stock, response_com)
  
  all_data <- combined %>% purrr::reduce(full_join, by = "Date")
  final_df <- all_data[complete.cases(all_data), ]
  
  dataframe_portfolio <- final_df[,c(3,6,9,12,14,16)]
  names(dataframe_portfolio) <- c('GE', 'Walmart',"Altria", "Brent", "Gold", "USD_CHF")
  
  save(final_df, file='./code/r_code/results_application/multivariate_models/prepared_data/final_df')
  save(dataframe_portfolio, file = '~/Desktop/Thesis/code/r_code/results_application/multivariate_models/prepared_data/dataframe_portfolio')
}

file_portfolio_weighted <- './code/r_code/results_application/multivariate_models/prepared_data/portfolio_returns'

if(file.exists(file_portfolio_weighted)) {
  load(file_portfolio_weighted)
} else {
  set.seed(42)
  weights <- matrix(rexp(ncol(dataframe_portfolio)*50), nrow=50, ncol=6)
  weights <- t(apply(weights, 1, FUN=function(x) x/sum(x)))
  
  portfolio_returns_matrix <- matrix(NA, nrow=2992, ncol=50)
  for (i in 1:50) {
    portfolio_returns_matrix[, i] <- rowSums(weights[i,]*dataframe_portfolio)
  }
  save(portfolio_returns_matrix, file='./code/r_code/results_application/multivariate_models/prepared_data/portfolio_returns_matrix')
  save(weights, file='./code/r_code/results_application/multivariate_models/weight_matrix')
}

#### Modelling ####

## SANN
# Lag portfolio returns up to desired level
sann_portfolio_models <- './code/r_code/results_application/multivariate_models/results_portfolio_model.csv'
if(file.exists(sann_portfolio_models)){
  print('Portfolio file already exists')
} else {
  max_lags <- 2
  lag_df_all <- matrix(ncol = 0, nrow = 2992)
  for(i in 1:6) {
    lag_er <- lag_timeseries_mv(dataframe_portfolio[,i], max_lags = max_lags)
    lag_df_all <- cbind(lag_df_all, lag_er)
  }
  
  # Run for all generated portfolios
  colnames_portfolio <- matrix(c('num_it', c(1:1000)), nrow=1,ncol=1001)
  write.table(colnames_portfolio, file = './code/r_code/results_application/multivariate_models/results_portfolio_model.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  for (i in 1:50) {
    
    # Add portfolio value and square
    df_var_estim <- cbind(portfolio_returns_matrix[i, ], lag_df_all)
    df_var_estim <- df_var_estim^2
    
    train <- 2992-1000
    prepared_keras <- get_keras_data(input_data = df_var_estim, train = train,
                                     position_linear = c(2:(max_lags*ncol(dataframe_portfolio))),
                                     position_nonlinear = c(2:(max_lags*ncol(dataframe_portfolio))),
                                     position_y = c(1), skip_first = 3)
    
    use_session_with_seed(42)
    nonlinear_input <- layer_input(shape = c(12), name='nonlin_input')
    nonlinear <- nonlinear_input %>%
      layer_dense(units =80, activation = 'relu' , activity_regularizer = regularizer_l1(l=0.00001)) %>%
      layer_dense(units = 5 , activation = 'relu')
    
    # Add linear layer and combine
    linear_input <- layer_input(shape = c(12), name = 'linear_input')
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
          epochs = 50, verbose = F)
    
    prediction_hybrid <- model_hybrid %>%
      predict(list(nonlin_input=prepared_keras[[2]][[3]], linear_input=prepared_keras[[2]][[2]]))
    
    pred_storeready <- matrix(c(i, prediction_hybrid), nrow=1, ncol=1001)
    
    write.table(pred_storeready, file = './code/r_code/results_application/multivariate_models/results_portfolio_model.csv',
                sep=";", append=TRUE,
                row.names = FALSE, col.names = FALSE)
    
  }
  
}

## MGARCH model

mgarch_results_file <- './code/r_code/results_application/multivariate_models/predictions_mgarch'
if(file.exists(mgarch_results_file)) {
  print('MGARCH model was already generated')
} else {
  num_cores <- detectCores()
  
  mgarch_specification = ugarchspec(mean.model = list(armaOrder = c(1, 1)),
                                    variance.model = list(garchOrder = c(1,1),
                                                          model = 'sGARCH'),
                                    distribution.model = 'norm')
  
  # Replicate for all columns
  all_specs = multispec(replicate(6,
                                  mgarch_specification))
  
  # Specify MGARCH model (DCC)
  specgarch = dccspec(uspec = all_specs,
                      dccOrder = c(1, 1),
                      distribution = 'mvnorm')
  
  # Run on local cluster
  out_sample<- 1000
  n <- nrow(dataframe_portfolio)
  
  cl = makePSOCKcluster(num_cores)
  multf = multifit(all_specs, dataframe_portfolio[1:(n-out_sample),], cluster=cl)
  fit_mgarch = dccfit(specgarch, data = dataframe_portfolio, fit.control = list(eval.se = TRUE), fit = multf,
                      out.sample = out_sample, cluster=cl)
  stopCluster(cl)
  
  forecast_class <- dccforecast(fit = fit_mgarch,
                                n.ahead = 1,
                                n.roll=(out_sample-1))
  covariance_forecast <- rcov(forecast_class)
  correlation_forecast <- rcor(forecast_class)
  
  save(covariance_forecast, file='./code/r_code/results_application/multivariate_models/predictions_mgarch')
  save(correlation_forecast, file='./code/r_code/results_application/multivariate_models/correlation_prediction_mgarch')
}