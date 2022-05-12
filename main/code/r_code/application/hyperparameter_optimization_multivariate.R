##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Hyperparameter optimization for 
##         Multivariate CAViaR models
## Tested: Apple-Darwin
##----------------------------------------------##

# Three notes on the code:
# - we will use the first 200 observations of the training data to optimize the algorithm, 
#   this choice is somewhat random and proper cross validation would be better, but given the 
#   computing resources available we will opt for this approach.
# - We will optimize the models for the 99th quantile. This is apt if we assume symmetric returns 
#   (which arguably is not a very good assumption). Nevertheless, it is necessary because we 
#   model the variance, which by definition is always positive.
# - We used the same procedure for the univariate case
rm(list=ls())
gc()

setwd('~/Documents/Backup_masterarbeit_falls_ich_wieder_git_verhaue/public_github_push/Thesis/')
library(keras)
source('./code/r_code/helper_functions/result_transformation.R')

## We will optimize the model a portfolio with equally weighted assets

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

# Create equal portfolio weights
weights <- matrix(rep(1/6, 6), nrow=1)
portfolio_returns <- rowSums(weights*dataframe_portfolio)

# Create keras dataframes
lag_df_all <- matrix(ncol = 0, nrow = nrow(dataframe_portfolio))
for(i in 1:dim(dataframe_portfolio)[2]) {
  lag_er <- lag_timeseries_mv(dataframe_portfolio[,i], max_lags = 2)
  lag_df_all <- cbind(lag_df_all, lag_er)
}

df_var_estim <- cbind(portfolio_returns, lag_df_all)
df_var_estim <- df_var_estim^2

train <- nrow(dataframe_portfolio) - 1000
prepared_keras <- get_keras_data(input_data = df_var_estim, train = train,
                                 position_linear = c(2:13), position_nonlinear = c(2:13),
                                 position_y = c(1), skip_first = 3)

# Initialize search grid
l_1_values_1 <- c(1e-04, 1e-5, 1e-6, 1e-06)
optimization_x <- prepared_keras[[1]][[3]][201:1989,]
optimization_y <- prepared_keras[[1]][[1]][201:1989,]
optimization_hold_out_x <- prepared_keras[[1]][[3]][1:200,]

# Run Evaluation
evaluation_matrix <- matrix(NA, ncol=1, nrow = length(l_1_values_1))
for (i in 1:length(l_1_values_1)) {
  
  l_1_pen <- l_1_values_1[i]
  use_session_with_seed(42)
  nonlinear_input <- layer_input(shape = c(12), name='nonlin_input')
  nonlinear <- nonlinear_input %>%
    layer_dense(units =80, activation = 'relu' , activity_regularizer = regularizer_l1(l=l_1_pen)) %>%
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
  get_weights(model_hybrid)
  # Fit Hybrid
  model_hybrid %>%
    fit(x= list(nonlin_input=optimization_x, linear_input=optimization_x),
        epochs = 30, verbose = F)
  
  prediction_hybrid <- model_hybrid %>%
    predict(list(nonlin_input=optimization_hold_out_x, linear_input=optimization_hold_out_x))
  
  terms <- exceeded_estimation(-sqrt(prediction_hybrid),portfolio_returns[1:200], 0.01)
  evaluation_matrix[i,] <- unname(terms[3,]) 
}

use_which <- max(which(evaluation_matrix==1))
l_1_value <- l_1_values_1[use_which]
second_iteration <- c(l_1_value, 0.5*l_1_value)

eval_mat_second_it <- matrix(NA, ncol=1, nrow = length(second_iteration))
for (i in 1:length(second_iteration)) {
  
  l_1_pen <- second_iteration[i]
  use_session_with_seed(42)
  nonlinear_input <- layer_input(shape = c(12), name='nonlin_input')
  nonlinear <- nonlinear_input %>%
    layer_dense(units =80, activation = 'relu' , activity_regularizer = regularizer_l1(l=l_1_pen)) %>%
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
    fit(x= list(nonlin_input=optimization_x, linear_input=optimization_x),
        y=optimization_y, steps_per_epoch = 200,
        epochs = 30, verbose = F)
  
  prediction_hybrid <- model_hybrid %>%
    predict(list(nonlin_input=optimization_hold_out_x, linear_input=optimization_hold_out_x))
  
  terms <- exceeded_estimation(-sqrt(prediction_hybrid),portfolio_returns[1:200], 0.01)
  eval_mat_second_it[i,] <- unname(terms[3,]) 
}

final_param <- max(which(eval_mat_second_it==1))
print(paste('Final penalization parameter is: ', final_param))