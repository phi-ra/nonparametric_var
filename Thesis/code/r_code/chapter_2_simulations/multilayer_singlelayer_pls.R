##----------------------------------------##
##  Author: Philipp Ratz
##  Title:  Illustration Multi- and 
##          Single layer ANN
##  Year:   2019
##  Tested: apple-darwin
##-----------------------------------------##
rm(list=ls())
gc()

# Load Packages, data and helperfunctions
library(dplyr)
library(reshape2)
library(keras)
library(pls)

source('./code/r_code/helper_functions/data_generation.R')

#### Simulation ####
history_file <- './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/history_single'

if(file.exists(history_file)) {
  load(file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/history_single')
  load(file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/history_multi')
  load(file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/history_large')
  load(file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/prediction_single')
  load(file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/prediction_multi')
  load(file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/prediction_large')
} else {
  
  set.seed(42)
  simulation_data <- generate_data_iid(size=400, mean=0, sd=441,
                                       x_from=-10, x_to=10,
                                       break_at = 1)
  use_session_with_seed(42)
  single_layer <- keras_model_sequential() %>%
    layer_dense(units = 100, activation = "sigmoid", input_shape = c(1)) %>%
    layer_dense(units=1, activation = 'linear')
  
  single_layer %>% compile(
    optimizer = optimizer_adam(lr=0.02),
    loss = "mse"
  )
  
  history_single <- single_layer %>%
    fit(x=simulation_data[[1]], y=simulation_data[[2]],
        epochs = 250, verbose = T,steps_per_epoch = 100, shuffle = F)
  
  prediction_single <- single_layer %>%
    predict(simulation_data[[1]])
  
  use_session_with_seed(42)
  multi_layer <- keras_model_sequential() %>%
    layer_dense(units = 100, activation = 'sigmoid', input_shape = c(1)) %>%
    layer_dense(units = 10, activation = 'relu') %>%
    layer_dense(units=1, activation = 'linear')
  
  multi_layer %>% compile(
    optimizer = optimizer_adam(lr=0.02),
    loss = "mse"
  )
  
  history_multi <- multi_layer %>%
    fit(x=simulation_data[[1]], y=simulation_data[[2]],
        epochs = 300, verbose = T,steps_per_epoch = 100, shuffle = F)
  
  prediction_multi <- multi_layer %>%
    predict(simulation_data[[1]])
  
  use_session_with_seed(42)
  single_large <- keras_model_sequential() %>%
    layer_dense(units = 408, activation = "sigmoid", input_shape = c(1)) %>%
    layer_dense(units=1, activation = 'linear')
  
  single_large %>% compile(
    optimizer = optimizer_adam(lr=0.02),
    loss = "mse"
  )
  
  history_large <- single_large %>%
    fit(x=simulation_data[[1]], y=simulation_data[[2]],
        epochs = 300, verbose = T,steps_per_epoch = 100, shuffle = F)
  
  prediction_large <- single_large %>%
    predict(simulation_data[[1]])
  
  ## Store Results
  save(history_single, file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/history_single')
  save(history_multi, file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/history_multi')
  save(history_large, file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/history_large')
  save(prediction_single, file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/prediction_single')
  save(prediction_multi, file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/prediction_multi')
  save(prediction_large, file = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/prediction_large')
}
#### Illustration ####
set.seed(42)
simulation_data <- generate_data_iid(size=400, mean=0, sd=441,
                                     x_from=-10, x_to=10,
                                     break_at = 1)

mse_single <- mean((simulation_data[[3]] - prediction_single)^2)
mse_multi <- mean((simulation_data[[3]] - prediction_multi)^2)
mse_large <- mean((simulation_data[[3]] - prediction_large)^2)

## Fitting
png('./tex_files/thesis/Figures/r_figures/visualisation_multilayer_mse', 
    width = 1192, height = 414)
par(mfrow=c(1,3))
plot(simulation_data[[1]], simulation_data[[2]], main='Single 100 Units',
     xlab = paste('MSE:', round(mse_single)), ylab="")
lines(simulation_data[[1]], simulation_data[[3]], col='red', lwd=4)
lines(simulation_data[[1]], prediction_single, col='blue', lwd=3)

plot(simulation_data[[1]], simulation_data[[2]], main='Multilayer',
     xlab=paste('MSE:', round(mse_multi)), ylab="")
lines(simulation_data[[1]], simulation_data[[3]], col='red', lwd=4)
lines(simulation_data[[1]], prediction_multi, col='blue', lwd=3)

plot(simulation_data[[1]], simulation_data[[2]], main='Single 408 Units',
     xlab=paste('MSE:', round(mse_large)), ylab="")
lines(simulation_data[[1]], simulation_data[[3]], col='red', lwd=4)
lines(simulation_data[[1]], prediction_large, col='blue', lwd=3)
dev.off()


png('./tex_files/thesis/Figures/r_figures/history_multilayer_comparison.png', 
    width = 1228, height = 410)
par(mfrow=c(1,3))
plot(history_single$metrics$loss, ylab="", xlab='Epochs', ylim = c(1.5e5, 1e6), 
     main='Singlelayer')
plot(history_multi$metrics$loss, ylab="", xlab='Epochs', ylim = c(1.5e5, 1e6),
     main='Multilayer')
plot(history_large$metrics$loss, ylab="", xlab='Epochs', ylim = c(1.5e5, 1e6),
     main='Singlelayer-Large')
dev.off()

## Illustration with PlS
network_linear_second <- './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/keras_pls.h5'
if(file.exists(network_linear_second)) {
  pls_network <- load_model_hdf5(filepath = network_linear_second)
  load('./code/r_code/results_simulation/results_bias_variance/comparison_overfitting/results_keras_pls')
} else {
  
  # Fit model with second linear layer
  use_session_with_seed(42)
  pls_network <- keras_model_sequential() %>%
    layer_dense(units = 100, activation = "sigmoid", input_shape = c(1)) %>%
    layer_dense(units= 10 , activation = 'linear') %>%
    layer_dense(units=1, activation = 'linear')
  
  pls_network %>% compile(
    optimizer = optimizer_adam(),
    loss = "mse"
  )
  
  history_pls <- pls_network %>%
    fit(x=simulation_data[[1]], y=simulation_data[[2]],
        epochs = 250, verbose = T,steps_per_epoch = 100, shuffle = F)
  
  prediction_pls_net <- pls_network %>%
    predict(simulation_data[[1]])
  
  #Store results
  save_model_hdf5(pls_network, filepath = './code/r_code/results_simulation/results_bias_variance/comparison_overfitting/keras_pls.h5')
  save(prediction_pls_net, file='./code/r_code/results_simulation/results_bias_variance/comparison_overfitting/results_keras_pls')
}

## Access output of hidden layer
layer_1 <- 'dense_1'
setup_output <- keras_model(inputs = pls_network$input,
                            outputs = get_layer(pls_network, layer_1)$output)
output_layer_1 <- as.data.frame(predict(setup_output, simulation_data[[1]]))

df_pls <- data.frame(cbind(simulation_data[[2]], as.matrix(output_layer_1)))
names(df_pls) <- c('y', paste('v_', c(1:100), sep=''))

formula_pls <- formula(paste('y~', paste('v_', c(1:100), sep='', collapse = '+')))
pls_reg <- plsr(formula_pls,
                data=df_pls, ncomp=20)
predict_pls <- predict(pls_reg, df_pls[,2:101], ncomp = 10)


png('./tex_files/thesis/Figures/r_figures/ann_pls_comparison.png', 
    width = 700, height = 536)
plot(simulation_data[[1]], simulation_data[[2]], main='Fit Comparison - ANN and PLS',
     xlab = '', ylab = '', col='dimgrey')
lines(simulation_data[[1]], prediction_pls_net, col='red', lwd=4)
lines(simulation_data[[1]], predict_pls, col='black', lwd=3, lty='dotted')
legend('bottomright', legend = c('ANN', 'PLS'), lty=c('solid', 'dotted'), lwd=c(2,2), col=c('red', 'black'))
dev.off()