##-----------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Autoregressive chaos model - 
##         approximation with increasing
##         number of hidden units. 
## Tested: Linux Mint
##-----------------------------------------##
rm(list=ls())
gc()

# Load Packages
library(keras)
library(parallel)
source('./code/r_code/helper_functions/data_generation.R')

file_mse_per_units <- './code/r_code/results_simulation/mse_per_units/mse_per_units.csv'

if(file.exists(file_mse_per_units)) {
  results_mse <- read.csv('./results_simulation/mse_per_units/mse_per_units.csv',
                          sep=';')
} else {
  print('Starting simulation process')
  # Function parameters and helpers
  N <- 600
  lags <- 4
  series_length <- N + lags
  n_hidden_units <- c(seq(1,25,1), seq(26,120,2))
  n_cores <- detectCores()
  
  # Data generation
  # Initialize first observation
  set.seed(42)
  y_0 <- runif(1)
  
  # Initialize observation vector
  y_vec <- numeric(N)
  y_vec[1] <- y_0
  
  # Loop through vector
  for (pos in 2:length(y_vec)) {
    y_vec[pos] <- 0.3*y_vec[pos-1] + (22/pi)*sin(2*y_vec[(pos-1)] + 1/3)
  }
  
  # Initialize "results-dataframe"
  col_names <- c('hidden_units', 'MSE')
  
  write.table(col_names, file = './results_simulation/mse_per_units/mse_per_units.csv',
              sep=";",
              row.names = FALSE, col.names = FALSE)
  
  data_train <- get_lagged_data(lags = lags, input_data = y_vec)
  
  for (num_unit in n_hidden_units) {
    
    network <- keras_model_sequential() %>%
      layer_dense(units = num_unit, activation = "sigmoid", input_shape = c(4)) %>% 
      layer_dense(units = 1, activation = 'linear')
    
    network %>% compile(
      optimizer = optimizer_adam(),
      loss = "mse"
    )
    
    network %>%
      fit(data_train[[1]], data_train[[2]],
          epochs = 200, steps_per_epoch = 100,
          workers=n_cores, shuffle = F, verbose = T)
    
    y_hat <- network %>%
      predict(data_train[[1]])
    
    mse <- mean((y_hat-data_train[[2]][1:596])^2)
    results <- c(num_unit, mse)
    
    write.table(results, file = './code/r_code/results_simulation/mse_per_units/mse_per_units.csv',
                sep=";",append = TRUE,
                row.names = FALSE, col.names = FALSE)
    rm(network, results, mse, y_hat)
    gc()
    print(paste('finished iteration', num_unit))
  }
  results_mse <- read.csv('./code/r_code/results_simulation/mse_per_units/mse_per_units.csv',
                          sep=';')
}

#### Visualisation ####

png('./tex_files/thesis/Figures/r_figures/mse_per_units_plot.png', 
    width = 540, height = 408)
x <- results_mse$hidden_units
y <- results_mse$MSE
lo <- loess(y~x)
plot(x,y, xlim=c(0,120),
     pch=19,
     ylab='MSE', xlab='Hidden Units')
lines(results_mse$hidden_units, predict(lo), col='blue', lwd=2.5)
dev.off()
