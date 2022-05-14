##----------------------------------------------##
## Author: Philipp Ratz
##         UQAM
## Year:   2022
## Title:  CAViaR parametric models
## Tested: Apple Metal
##----------------------------------------------##
library(econophysics)
library(tidyverse)
library(quantreg)
library(zoo)
library(rugarch)
library(sfsmisc)

source('./code/r_code/helper_functions/data_generation.R')
source('./code/r_code/helper_functions/result_transformation.R')

#### Univariate ####
data_ge <- read.csv(file = './data/stocks/ge.us.txt')
prepared_ge <- prepare_returns(data_ge, scale = TRUE,
                               date_from = '2000-01-01', date_to = '2012-12-31',
                               log_corr = TRUE)

# Run some dataprep for the "features"
prepared_ge %>% 
  as_tibble() %>% 
  mutate(tot_ret = c(prepared_ge$returns)) %>% 
  select(Date, returns = tot_ret) %>% 
  mutate(lag_return_1 = lag(returns, n = 1), 
         lag_return_2 = lag(returns, n = 2)) %>% 
  drop_na() -> all_obs_univariate

all_obs_univariate %>% head(2262) -> train 
all_obs_univariate %>% filter(!Date %in% train$Date) -> test

caviar_sav <- econophysics::caviarOptim(train$returns,
                                        model = 1,
                                        pval=0.99)

caviar_sav$bestPar


## Predictions Univariate
results_caviar_univ <- matrix(nrow = length(test$returns), ncol=2)

# create initial params
# and at time t available return vector
var_pred_ <- caviar_sav$VarPredict
lagged_returns <- c(train$returns[length(train$returns)] ,test$returns)

for (time_ in c(1:length(test$returns))) {
  results_caviar_univ[time_, 1] <- var_pred_
  results_caviar_univ[time_, 2] <- if_else(test$returns[time_] < var_pred_ ,
                                           1, 0)
  
  var_pred_ <- (caviar_sav$bestPar[1] +
                 caviar_sav$bestPar[2]*var_pred_ +
                 caviar_sav$bestPar[3]*abs(lagged_returns[time_]))
}
results_caviar_univ[2:1001]

caviar_number_p_01 <- VaRTest(test$returns[1:1001], results_caviar_univ[1:1001], alpha = 0.01)
caviar_duration_p_01 <- VaRDurTest(test$returns[1:1001], results_caviar_univ[1:1001], alpha = 0.01)

caviar_number_p_01$actual.exceed
caviar_number_p_01$uc.LRp
caviar_duration_p_01$LRp

uncond_quantiles <- rep(-unname(quantile(train$returns, 0.01)), 1001)
int_caviar <- integrate.xy(x=seq(1,1001,1), c(-results_caviar_univ[1:1001]))
int_var_uncond <- integrate.xy(x = seq(1,1001,1), uncond_quantiles)

((int_caviar - int_var_uncond)/int_var_uncond)*100

plot(test$returns, type='l')
lines(results_caviar_univ[1:1001], col='red')


#### Portfolio ####
# Load data
data_ge <- read.csv(file = './data/stocks/ge.us.txt')
data_walmart <- read.csv(file = './data/stocks/wmt.us.txt')
data_mo <- read.csv(file = './data/stocks/mo.us.txt')

# Commodities and exchange
data_brent <- read.csv(file = './data/commodities/brent_crude.csv',
                       na.strings = '.')
data_gold <- read.csv(file = './data/commodities/gold_price.csv',
                      na.strings = '.')
data_chf <- read.csv(file = './data/commodities/usd_chf_exchange.csv', 
                     na.strings = '.')


stock_list <- list(data_ge, data_walmart, data_mo)
response_stock <- lapply(stock_list,function(x) prepare_returns(x,
                                                                scale = TRUE,
                                                                date_from = '2000-01-01',
                                                                date_to = '2012-12-31'))

commodities_list <- list(data_brent, data_gold, data_chf)
response_com <- lapply(commodities_list,
                       function(x) prepare_returns_commodities(x, scale = TRUE,
                                                               date_from = '2000-01-01',
                                                               date_to = '2012-12-31'))

combined <- c(response_stock, response_com)

all_data <- combined %>% purrr::reduce(full_join, by = "Date")
final_df <- all_data[complete.cases(all_data), ]

dataframe_portfolio <- final_df[,c(8, 15, 22, 24, 26, 28 )]
names(dataframe_portfolio) <- c('GE', 'Walmart',"Altria", "Brent", "Gold", "USD_CHF")

set.seed(42)
weights <- matrix(rexp(ncol(dataframe_portfolio)*50), nrow=50, ncol=6)
weights <- t(apply(weights, 1, FUN=function(x) x/sum(x)))

portfolio_returns_matrix <- matrix(NA, nrow=2992, ncol=50)
for (i in 1:50) {
  portfolio_returns_matrix[, i] <- rowSums(weights[i,]*dataframe_portfolio)
}

# Run CAViaR for every sub-portfolio
results_matrix = matrix(nrow = 50, ncol=5)

portfolio_returns_matrix[1:1992, ] -> train_mat
portfolio_returns_matrix[1993:dim(portfolio_returns_matrix)[1], ] -> test_mat

for (port_num in c(1:50)){
  
  caviar_sav <- econophysics::caviarOptim(train_mat[, port_num],
                                          model = 1,
                                          pval=0.99)
  
  results_matrix[port_num, 1] <- caviar_sav$bestPar[2]
  
  # Predict
  results_caviar_tmp <- matrix(nrow = length(test_mat[, port_num]), ncol=2)
  var_pred_ <- caviar_sav$VarPredict
  lagged_returns <- c(train_mat[length(train_mat[, port_num]), port_num],
                      test_mat[, port_num])
  
  for (time_ in c(1:length(test_mat[, port_num]))) {
    results_caviar_tmp[time_, 1] <- var_pred_
    results_caviar_tmp[time_, 2] <- if_else(test_mat[time_, port_num] < var_pred_ ,
                                             1, 0)
    
    var_pred_ <- (caviar_sav$bestPar[1] +
                    caviar_sav$bestPar[2]*var_pred_ +
                    caviar_sav$bestPar[3]*abs(lagged_returns[time_]))
  }
  
  caviar_number_p_01 <- VaRTest(test_mat[, port_num], results_caviar_tmp[,1], alpha = 0.01)
  caviar_duration_p_01 <- VaRDurTest(test_mat[, port_num], results_caviar_tmp[,1], alpha = 0.01)
  
  results_matrix[port_num, 2] <- caviar_number_p_01$actual.exceed
  results_matrix[port_num, 3] <- if_else(caviar_number_p_01$uc.LRp > 0.05, 
                                         0, 1)
  results_matrix[port_num, 4] <- if_else(caviar_duration_p_01$LRp > 0.05, 
                                         0, 1)
  
  uncond_quantiles <- rep(-unname(quantile(train_mat[, port_num], 0.01)), 1000)
  int_caviar <- integrate.xy(x=seq(1,1000,1), c(-results_caviar_tmp[,1]))
  int_var_uncond <- integrate.xy(x = seq(1,1000,1), uncond_quantiles)
  
  results_matrix[port_num, 5] <- ((int_caviar - int_var_uncond)/int_var_uncond)*100
  
  print(port_num)
}

results_matrix[, 1] %>% mean()
results_matrix[, 2] %>% mean()
results_matrix[,3] %>% mean()
results_matrix[,4] %>% mean()
results_matrix[,5] %>% mean()
