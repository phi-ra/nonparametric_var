##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Visualisations for the multivariate
##         ARCH-Models
## Tested: Apple-Darwin
##----------------------------------------------##
rm(list=ls())
gc()

library(sn)
library(MASS)
library(stargazer)
library(reshape2)
library(keras)

source('./code/r_code/helper_functions/data_generation.R')

## Read in Data
tryCatch( {
  # Visualisation
  # Skewed-t  
  true_process_nonnorm <- read.csv('./code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/omega_true.csv',
                               sep=';', header=F)
  sann_visual_nonnorm <- read.csv('./code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/sann_pred.csv',
                                  sep=';', header=F)
  garch_visual_nonorm <- read.csv('./code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/garch_pred.csv',
                                  sep=';', header=F)
  # Nonlinear form 
  path_string <- paste(getwd(), '/code/r_code/results_simulation/results_multivariate_timeseries/form_misspecified/saved_models',
                       sep='')
  load(paste(path_string, '/error_matrix', sep = ''))
  load(paste(path_string, '/sigma_data', sep = ''))
  load(paste(path_string, '/hybrid_prediction', sep = ''))
  load(paste(path_string, '/mgarch_prediction', sep = ''))
  load(paste(path_string, '/hybrid_prediction', sep = ''))
  hybrid_model <- load_model_hdf5(paste(path_string, '/hybrid_model.h5', sep=''))
  
  # Numeric Results
  nonlinear_results <- read.csv('./code/r_code/results_simulation/results_multivariate_timeseries/form_misspecified/variance_nonlinear.csv',
                                sep=';')
  covariance_nonnormal <- read.csv('./code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/covariance_nonnormal.csv',
                                   sep=';')
  variance_nonnormal <- read.csv('./code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/variance_nonnormal.csv',
                                 sep=';')
  variance_normal <- read.csv('./code/r_code/results_simulation/results_multivariate_timeseries/distribution_misspecified/variance_normal.csv',
                              sep=';')
  
},
error=function(e) {
  print('Failed to load necessary data, have you generated all of it? You need to run all multivariate generating files first')
})

#### Visualisation ####
## Nonlinear form
i=1
png('./tex_files/thesis/Figures/r_figures/fig_ann_vs_mgarch_nonlinear.png', 
    width = 838, height = 575)
plot(sigma_data[901:1000, i], type='l', lwd=2, ylab="", xlab='1 Step ahead forecasts', ylim=c(0.3,1))
correction_bias_sann <- mean(sigma_data[701:800, i]) - mean(prediction_hybrid[,i])
correction_bias_mgarch <- mean(sigma_data[801:900, i]) - mean(forecast_variance[,i,])
lines(prediction_hybrid[,1]+correction_bias_sann, col='red', lty='dashed', lwd=2)
lines(forecast_variance[,i,]+correction_bias_mgarch, col='green', lty='dashed', lwd=2)
mtext(expression(sigma[t]),side=2,las=1,line=1,
      adj = 2.5)
legend('bottomright',
       legend = c('Truth', 'SANN', 'DCC'),
       col=c('black', 'red', 'green'),
       lty=1:3,
       lwd=2)
dev.off()

## Distributional misspecification
png('./tex_files/thesis/Figures/r_figures/multivariate_fit_mgarch_vs_sann.png', 
    width = 838, height = 575)
plot(true_process_nonnorm[[1]], type='l', lwd=2, ylab="", xlab='1 Step ahead forecasts')
lines(sann_visual_nonnorm, col='red', lty='dashed', lwd=2)
lines(garch_visual_nonorm, col='green', lty='dashed', lwd=2)
mtext(expression(sigma[t]),side=2,las=1,line=1,
      adj = 2.5)
legend('topright',
       legend = c('Truth', 'ANN', 'DCC'),
       col=c('black', 'red', 'green'),
       lty=1:3,
       lwd=2)
dev.off()

## Simulate the error distribution
num_var = 10
skew_param = -1.5
nu = 10
burn_in = 100
n = 1000
n_samples = (burn_in + n)

## Draw from skewed distribution
set.seed(42)
xi <- runif(num_var)
Omega <- diag(num_var)
correlation_mat <- positive_matrix(num_var, ev=runif(num_var,-1,1))
Omega[which(row(Omega)!=col(Omega))] <- correlation_mat[which(row(correlation_mat)!=col(correlation_mat))]
alpha <- c(rep(skew_param,num_var))
error_matrix <- rmst(n_samples, xi, Omega, alpha, nu)

png('./tex_files/thesis/Figures/r_figures/error_distribution_skewed_t.png', 
    width = 664, height = 425)
plot(density(error_matrix[,1]), main='', ylab="", xlab="", lwd=2)
lines(density(rnorm(1000)), col='red', lwd=2, lty='dashed')
title('Distribution of errors from skewed t')
legend('topright',
       legend = c('Skewed-t', 'Normal'),
       col=c('black', 'red'),
       lty=1:2,
       lwd=2)
dev.off()

# Plot correlations of error terms (take 6:9) and one
# Currently not in the thesis but nice to have
dev.off()
par(mfrow=c(2,2))

plot(error_matrix[,1], error_matrix[,6],
     xlab="", ylab="")
mtext(expression(epsilon[6]),side=2,las=1,line=1,
      adj = 2.5)
mtext(expression(epsilon[1]),side=1,las=1,
      adj = 0.5, padj = 2.7)
title(bquote(rho ~ .(paste('=', round(cor(error_matrix[,1], error_matrix[,6]),3)))), line=1)

plot(error_matrix[,1], error_matrix[,7],
     xlab="", ylab="")
mtext(expression(epsilon[7]),side=2,las=1,line=1,
      adj = 2.5)
mtext(expression(epsilon[1]),side=1,las=1,
      adj = 0.5, padj = 2.7)
title(bquote(rho ~ .(paste('=', round(cor(error_matrix[,1], error_matrix[,7]),3)))), line=1)

plot(error_matrix[,1], error_matrix[,8],
     xlab="", ylab="")
mtext(expression(epsilon[8]),side=2,las=1,line=1,
      adj = 2.5)
mtext(expression(epsilon[1]),side=1,las=1,
      adj = 0.5, padj = 2.7)
title(bquote(rho ~ .(paste('=', round(cor(error_matrix[,1], error_matrix[,8]),3)))), line=1)

plot(error_matrix[,1], error_matrix[,9],
     xlab="", ylab="")
mtext(expression(epsilon[9]),side=2,las=1,line=1,
      adj = 2.5)
mtext(expression(epsilon[1]),side=1,las=1,
      adj = 0.5, padj = 2.7)
title(bquote(rho ~ .(paste('=', round(cor(error_matrix[,1], error_matrix[,9]),3)))), line=1)
dev.off()


#### Tables ####
results_normal <- matrix(round(colMeans(variance_normal[,c(3:5)]),3))
results_nonnormal <- matrix(round(colMeans(variance_nonnormal[,c(3:5)]),3))
results_nonormal_covariance <- matrix(round(colMeans(covariance_nonnormal[,c(3:5)]),3))
results_nonlinear_dist <- matrix(round(colMeans(nonlinear_results[,c(3:5)]),3))

rmspe_final <- cbind(results_normal, results_nonnormal, results_nonormal_covariance, results_nonlinear_dist)
# Let SANN be in the middle
rmspe_final <- rmspe_final[c(1,3,2),]
rownames(rmspe_final) <- c('M-ARCH', 'SANN', 'ANN')
colnames(rmspe_final) <- c('Normal', 'Nonnormal', 'NonnormalCov', 'Nonlinear')

stargazer(rmspe_final)
