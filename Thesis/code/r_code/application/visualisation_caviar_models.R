##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Visualisations of CAViaR Models
## Tested: Apple-Darwin
##----------------------------------------------##
rm(list=ls())
gc()

library(tidyverse)
library(rugarch)
library(sfsmisc)

#### Numeric Results ####
exceeded_estimation <- function(VaR_estimate, true_returns, alpha) {
  differences <- true_returns - VaR_estimate
  exceedances <- length(differences[which(differences < 0)])
  expected <- length(true_returns)*alpha
  ratio <- exceedances/expected
  final <- rbind(exceedances, expected, ratio)
  return(final)
}

tryCatch({
  # Load Data
  sann_predictions <- read.csv('./code/r_code/results_application/multivariate_models/results_portfolio_model.csv',
                               sep=';', header=TRUE)
  
  # Multivariate
  load('./code/r_code/results_application/multivariate_models/prepared_data/portfolio_returns_matrix')
  load('./code/r_code/results_application/multivariate_models/weight_matrix')
  load('./code/r_code/results_application/multivariate_models/predictions_mgarch')
  load('./code/r_code/results_application/multivariate_models/correlation_prediction_mgarch')
  load('./code/r_code/results_application/multivariate_models/prepared_data/dataframe_portfolio')
  load('./code/r_code/results_application/multivariate_models/prepared_data/final_df')
  
  # Univariate
  load('./code/r_code/results_application/univariate_models/table_univariate_results')
  load('./code/r_code/results_application/univariate_models/prepred_ge_data')
  load('./code/r_code/results_application/univariate_models/var_forecast_sann_rec')
  load('./code/r_code/results_application/univariate_models/var_forecast_garch')
  
  },
  error=function(e) {
    print('Failed to load necessary data, have you generated all of it? You need to run univariate_cavia_model.R and multivariate_caviar_models.R first')
    }
)

## Reweighting MGARCH Portfolio
weight_sigmas <- function(sigmas, weights) {
  apply(weights, 1, FUN = function(x) sqrt(x %*% sigmas %*% x))
}

weighted_list <- list()
for(i in 1:1000) {
  asset_correlation <- predictions_mgarch[[i]]*correlation_forecast[[i]]
  weighted_list[[i]] <- asset_correlation
}

# Transform Variables
weights <- as.matrix(weights)
weighted_portfolio_variance <- t(sapply(weighted_list,FUN = function(x) weight_sigmas(x[, , 1], weights)))

VaR_matrix_mgarch <- matrix(NA, nrow=1000, ncol=50)
for (i in 1:50) {
  VaR_matrix_mgarch[, i] <- weighted_portfolio_variance[, i] * qdist('std', 0.01, mu = 0, sigma = 1, shape = 100)
}

## Calculate Tests
Vartest_matrix_mgarch <- matrix(NA, nrow=50, ncol=1)
Vartest_matrix_sann <- matrix(NA, nrow=50, ncol=1)

exceedances_mgarch <- matrix(NA, nrow=50, ncol=1)
exceedances_sann <- matrix(NA, nrow=50, ncol=1)

for(i in 1:50) {
  test_mgarch <- VaRTest(alpha = 0.01, portfolio_returns_matrix[1993:2992, i], VaR_matrix_mgarch[,i], conf.level = 0.95)
  test_sann <- VaRTest(alpha = 0.01, portfolio_returns_matrix[1993:2992, i], -sqrt(sann_predictions[i,2:1001]), conf.level = 0.95)
  
  exceedances_mgarch[i,] <- test_mgarch$actual.exceed
  exceedances_sann[i, ] <- test_sann$actual.exceed
  
  Vartest_matrix_mgarch[i, ]  <- test_mgarch$cc.Decision
  Vartest_matrix_sann[i, ] <- test_sann$cc.Decision
}

Durtest_matrix_mgarch <- matrix(NA, nrow=50, ncol=1)
Durtest_matrix_sann <- matrix(NA, nrow=50, ncol=1)

for(i in 1:50) {
  dtest_mgarch <- VaRDurTest(alpha = 0.01, portfolio_returns[(2992-999):2992, i], VaR_matrix_mgarch[,i], conf.level = 0.95)
  dtest_sann <- VaRDurTest(alpha = 0.01, portfolio_returns[(2992-999):2992, i], -sqrt(sann_predictions[i,2:1001]), conf.level = 0.95)
  
  Durtest_matrix_mgarch[i, ]  <- dtest_mgarch$Decision
  Durtest_matrix_sann[i, ] <- dtest_sann$Decision
}

int_mgarch <- matrix(NA, nrow=50, ncol=1)
int_sann <- matrix(NA, nrow=50, ncol=1)
quantiles_port <- apply(portfolio_returns_matrix[1992:2992, ], 2, FUN=function(x) quantile(x, probs=c(0.01)))

for (i in 1:50){
  int_uncond <- integrate.xy(x=seq(1,1000,1), rep(-quantiles_port[i], 1000))
  int_var_garch <- integrate.xy(x = seq(1,1000,1), -VaR_matrix_mgarch[, i])
  int_var_sann <- integrate.xy(x = seq(1,1000,1), sqrt(sann_predictions[i,2:1001]))
  
  int_mgarch[i,] <- (int_var_garch - int_uncond)/int_uncond
  int_sann[i, ] <- (int_var_sann - int_uncond)/int_uncond
  
}

duration_test <- matrix(c(unname(table(Durtest_matrix_mgarch)[2]/50),
                          unname(table(Durtest_matrix_sann)[2]/50)),nrow=1)

number_test <- matrix(c(unname(table(Vartest_matrix_mgarch)[2]/50),
                        unname(table(Vartest_matrix_sann)[2]/50)), nrow=1)

exceedances <- matrix(c(mean(exceedances_mgarch), mean(exceedances_sann)), nrow=1)
int_var <- matrix(c(mean(int_mgarch)*100, mean(int_sann)*100), nrow=1)

results_multivariate <- rbind(exceedances, number_test,
                              duration_test, int_var)
colnames(results_multivariate) <- c('MGARCH(1,1)', 'SANN(2,0)')
rownames(results_multivariate) <- c('Exceedances', 'Failure Test',
                           'Duration Test', 'Change int VaR')

stargazer(results_multivariate, digits=2)
stargazer(results_univariate, digits = 2)

#### Summary stats and Appendix ####

summary_df <- final_df[,c(3,6,9,12,14,16)]
names(summary_df) <- c('GE', 'Walmart',"Altria", "Brent", "Gold", "USD/CHF")

stargazer(summary_df)
stargazer(cov(dataframe_portfolio))

# Equally weighted Portfolio

weights <- matrix(rep(1/6,6), nrow=1)
weights <- apply(weights, 1, FUN=function(x) x/sum(x))

portfolio_returns <- rowSums(weights*summary_df)
final_df$portfolio_returns <- portfolio_returns

png('./tex_files/thesis/Figures/r_figures/train_test_split_portfolio.png', 
    width = 639, height = 437)
plot(final_df$Date, final_df$portfolio_returns, type='l', xlab="", ylab='', main='Portfolio Log-Returns', col='white')
lines(final_df$Date[1:(2992-999)], final_df$portfolio_returns[1:(2992-999)])
lines(final_df$Date[(2992-999):2992], final_df$portfolio_returns[(2992-999):2992], col='darkorange3')
legend('bottomright',
       legend = c('Train', 'Test'),
       col=c('black', 'darkorange3'),
       lty=c(1,1),
       lwd=2)
dev.off()

png('./tex_files/thesis/Figures/r_figures/distribution_of_returns.png', 
    width = 639, height = 437)
plot(density(scale(final_df$portfolio_returns)), main='Distribution of log-returns')
lines(density(rnorm(10000)), col='red', lty='dashed')
legend('topright',
       legend = c('Returns', 'Normal'),
       col=c('black', 'red'),
       lty=c(1,2),
       lwd=2)
dev.off()


png('./tex_files/thesis/Figures/r_figures/returns_ge_stock.png', 
    width = 676, height = 412)
plot(prepared_ge$Date, prepared_ge$returns, col='white', ylab="", xlab="", main='Log-returns GE')
lines(prepared_ge$Date[1:(nrow(prepared_ge)-1000)], prepared_ge$returns[1:(nrow(prepared_ge)-1000)], col='black')
lines(prepared_ge$Date[(nrow(prepared_ge)-999):nrow(prepared_ge)], prepared_ge$returns[(nrow(prepared_ge)-999):nrow(prepared_ge)], col='darkorange')
legend('topright',
       legend = c('Training', 'Test'),
       col=c('black', 'darkorange'),
       lty=c(1,1),
       lwd=2)
dev.off()

png('./tex_files/thesis/Figures/r_figures/out_sample_fit_univariate.png', 
    width = 941, height = 551)
plot(prepared_ge$Date[(nrow(prepared_ge)-999):nrow(prepared_ge)],
     prepared_ge$returns[(nrow(prepared_ge)-999):nrow(prepared_ge)],
     type='l', ylab="", xlab="", ylim=c(-10,10))
lines(prepared_ge$Date[(nrow(prepared_ge)-999):nrow(prepared_ge)], var_forecast_garch, col='dodgerblue2')
lines(prepared_ge$Date[(nrow(prepared_ge)-999):nrow(prepared_ge)], var_forecast_hybrid_rec, col='lightsalmon3')
lines(prepared_ge$Date[(nrow(prepared_ge)-999):nrow(prepared_ge)],
      rep(unname(quantile(prepared_ge$returns[(nrow(prepared_ge)-999):nrow(prepared_ge)],
                          probs=0.01)),1000), col='red', lty='dashed')
legend('topright',
       legend = c('GARCH', 'SANN', 'Uncond. Quant.'),
       col=c('dodgerblue2', 'lightsalmon3', 'red'),
       lty=c(1,1,2),
       lwd=2)
dev.off()