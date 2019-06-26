##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Sources of overfitting
## Tested: Apple-Darwin/Linux-Mint
##----------------------------------------------##
rm(list=ls())
gc()

library(keras)
source('./code/r_code/helper_functions/data_generation.R')

file_overfitting_units <- './code/r_code/results_simulation/results_overfitting_behaviour/overfitting_with_units'

if(file.exists(file_overfitting_units)) {
  load(file_overfitting_units)
  set.seed(7)
  data_generation <- generate_data_iid(size = 400, mean=0, sd=331,
                                       x_from=-10, x_to = 10, break_at = 1)
  
} else {

  # Generate Data 
  set.seed(7)
  data_generation <- generate_data_iid(size = 400, mean=0, sd=331,
                                       x_from=-10, x_to = 10, break_at = 1)
  
  results_matrix <- matrix(c(1:401), nrow=1, ncol=401)
  
  hidden_units <- c(1, 150, 800)
  for (num_units in hidden_units) {
    
    use_session_with_seed(42)
    network <- keras_model_sequential() %>%
      layer_dense(units = num_units, activation = "sigmoid", input_shape = c(1)) %>% 
      layer_dense(units = 1, activation = 'linear')
    
    network %>% compile(
      optimizer = optimizer_adam(lr=0.02),
      loss = "mse"
    )
    
    network %>%
      fit(data_generation[[1]], data_generation[[2]],
          epochs = max(150, (num_units/3+100)), steps_per_epoch = 100,
          workers=n_cores, shuffle = F, verbose = T)
    
    y_hat <- network %>%
      predict(data_generation[[1]])
    
    y_hat_mat <- matrix(y_hat, nrow=1, ncol=401)
    
    results_matrix <- rbind(results_matrix , y_hat_mat)
    rm(list=setdiff(ls(), c('data_generation', 'results_matrix')))
    gc()
  }
  save(results_matrix, file = './code/r_code/results_simulation/results_overfitting_behaviour/overfitting_with_units')
}

png('./tex_files/thesis/Figures/r_figures/overfitting_with_units', 
    width = 887, height = 575)
plot(data_generation[[1]], data_generation[[2]], ylab="", xlab="x", axes=FALSE)
lines(data_generation[[1]], data_generation[[3]], lwd=2, col='black')
lines(data_generation[[1]], results_matrix[2,], lwd=2, lty='dotted', col='blue')
lines(data_generation[[1]], results_matrix[4,], lwd=2, col='red')
lines(data_generation[[1]], results_matrix[3,], lwd=2, col='green')
axis(1, labels = TRUE)
axis(2, labels=FALSE)
box()
legend('bottomright',
       legend = c('Truth', '1 Hidden Unit', '150 Hidden Units', '800 Hidden Units'),
       col=c('black', 'blue','green', 'red'),
       lty=c(1,3,1,1),
       lwd=2)
dev.off()

#### Overfitting with epochs ####

file_overfitting_epochs <- './code/r_code/results_simulation/results_overfitting_behaviour/overfitting_with_epochs'

if(file.exists(file_overfitting_epochs)) {
  load(file_overfitting_epochs)
  set.seed(7)
  data_generation <- generate_data_iid(size = 400, mean=0, sd=331,
                                       x_from=-10, x_to = 10, break_at = 1)
  
} else {
  
  epoch_list <- c(10, 100, 1000)
  results_matrix_passes <- matrix(c(1:401), nrow=1, ncol=401)
  
  for (num_passes in epoch_list) {
    
    use_session_with_seed(42)
    network <- keras_model_sequential() %>%
      layer_dense(units = 800, activation = "sigmoid", input_shape = c(1)) %>% 
      layer_dense(units = 1, activation = 'linear')
    
    network %>% compile(
      optimizer = optimizer_adam(lr=0.02),
      loss = "mse"
    )
    
    network %>%
      fit(data_generation[[1]], data_generation[[2]],
          epochs = num_passes, steps_per_epoch = 100,
          workers=n_cores, shuffle = F, verbose = T)
    
    y_hat <- network %>%
      predict(data_generation[[1]])
    
    y_hat_mat <- matrix(y_hat, nrow=1, ncol=401)
  
    results_matrix_passes <- rbind(results_matrix_passes , y_hat_mat)
    rm(list=setdiff(ls(), c('data_generation', 'results_matrix_passes')))
    gc()
  }
  save(results_matrix_passes, file = './code/r_code/results_simulation/results_overfitting_behaviour/overfitting_with_epochs')
}

png('./tex_files/thesis/Figures/r_figures/overfitting_with_epochs.png', 
    width = 887, height = 575)
plot(data_generation[[1]], data_generation[[2]], ylab="", xlab="x", axes=FALSE)
lines(data_generation[[1]], data_generation[[3]], lwd=2, col='black')
lines(data_generation[[1]], results_matrix_passes[2,], lwd=2, lty='dotted', col='blue')
lines(data_generation[[1]], results_matrix_passes[4,], lwd=2, col='red')
lines(data_generation[[1]], results_matrix_passes[3,], lwd=2, col='green')
axis(1, labels = TRUE)
axis(2, labels=FALSE)
box()
legend('bottomright',
       legend = c('Truth', '10 Epochs', '100 Epochs', '1000 Epochs'),
       col=c('black', 'blue','green', 'red'),
       lty=c(1,3,1,1),
       lwd=2)
dev.off()
