##-----------------------------------------##
##  Author: Philipp Ratz
##  Title:  Calculations for Bias-Variance
##          Tradeoffs
##  Year:   2019
##  Tested: Linux Mint
##-----------------------------------------##
rm(list=ls())
gc()

library(keras)
library(parallel)
source('./code/r_code/helper_functions/data_generation.R')


#### Tradeoff Figure ####
file_singlelayer_tradeoff <- './code/r_code/results_simulation/results_bias_variance/results_singlelayer/tradeoff_data_single.csv'

if(file.exists(file_singlelayer_tradeoff)) {
  print('File are already generated')
} else {
  results_col_names <- matrix(c('num_units', c(1:401)), nrow = 1, ncol = 402)
  write.table(results_col_names, file = './code/r_code/results_simulation/results_bias_variance/results_singlelayer/tradeoff_data_single.csv',
              sep=";",
              row.names = FALSE, col.names = FALSE)
  
  num_units <- c(1,seq(10,730,30))
  
  for(num_unit in num_units){
    ncores <- detectCores()
    sd=441
    mean=0
    size=400
    set.seed(42)
    simulation_data  <- generate_data_iid_seed(size = 400, mean=0, sd=441,
                                               x_from=-10, x_to = 10, break_at = 1)
    
    set.seed(42)
    noise <- rnorm(size+1, mean=mean, sd=sd)
    simulation_data[[2]] <- simulation_data[[3]]+noise
    
    
    network <- keras_model_sequential() %>%
      layer_dense(units = num_unit, activation = "sigmoid", input_shape = c(1)) %>%
      layer_dense(units = 1, activation = "linear")
    
    network %>% compile(
      optimizer = optimizer_adam(lr=0.02),
      loss = "mse"
    )
    
    network %>% fit(simulation_data[[1]], simulation_data[[2]],
                    epochs = (600 + num_unit),steps_per_epoch = 100,
                    workers=ncores, shuffle = T, verbose = F)
    
    y_hat <- network %>%
      predict(simulation_data[[1]])
    
    y_hat <- matrix(c(num_unit, y_hat), ncol=402, nrow=1)
    
    write.table(y_hat, file = './code/r_code/results_simulation/results_bias_variance/results_singlelayer/tradeoff_data_single.csv',
                sep=";",append = TRUE,
                row.names = FALSE, col.names = FALSE)

    print(paste('finished iteration',i))
    rm(list=(setdiff(ls(), c('generate_data_iid_seed', 'i', 'num_units', 'num_unit'))))
    gc()
  }
}

#### Comparison Single Multilayer ####
rm(list=ls())
gc()

file_comparison_layers <- './code/r_code/results_simulation/results_bias_variance/results_singlelayer/singlelayer_fixed_units.csv'

if(file.exists(file_comparison_layers)) {
  print('File are already generated')
} else {
  
  source('./code/r_code/helper_functions/data_generation.R')
  
  results_col_names <- matrix(c(1:401), nrow = 1, ncol = 401)
  write.table(results_col_names, file = './code/r_code/results_simulation/results_bias_variance/results_multilayer/multilayer_bias_variance_tradeoff.csv',
              sep=";",
              row.names = FALSE, col.names = FALSE)
  
  write.table(results_col_names, file = './code/r_code/results_simulation/results_bias_variance/results_singlelayer/singlelayer_bias_variance_tradeoff.csv',
              sep=";",
              row.names = FALSE, col.names = FALSE)
  
  for (i in 1:50) {
    sd=441
    mean=0
    size=400
    
    set.seed(42)
    simulation_data  <- generate_data_iid_seed(size = 400, mean=0, sd=441,
                                          x_from=-10, x_to = 10, break_at = 1)
    
    set.seed(i)
    noise <- rnorm(size+1, mean=mean, sd=sd)
    simulation_data[[2]] <- simulation_data[[3]]+noise
    
    
    network <- keras_model_sequential() %>%
      layer_dense(units = 250, activation = "sigmoid", input_shape = c(1)) %>%
      layer_dense(units = 1, activation = "linear")
    
    network %>% compile(
      optimizer = optimizer_adam(lr=0.02),
      loss = "mse"
    )
    
    network %>% fit(simulation_data[[1]], simulation_data[[2]],
                    epochs = 400,steps_per_epoch = 100,
                    workers=4, shuffle = T, verbose = T)
    
    y_hat <- network %>%
      predict(simulation_data[[1]])
    y_hat <- matrix(y_hat, ncol=401, nrow=1)
    
    write.table(y_hat, file = './code/r_code/results_simulation/results_bias_variance/results_singlelayer/singlelayer_bias_variance_tradeoff.csv',
                sep=";",append = TRUE,
                row.names = FALSE, col.names = FALSE)
    
    multilayer_version <- keras_model_sequential() %>%
      layer_dense(units=250, activation = 'sigmoid', input_shape = c(1)) %>%
      layer_dense(units=5, activation = 'linear') %>%
      layer_dense(units=1, activation= 'linear')
    
    multilayer_version %>% compile(
      optimizer = optimizer_adam(lr=0.02),
      loss = "mse"
    )
    
    multilayer_version %>% fit(simulation_data[[1]], simulation_data[[2]],
                               epochs = 400,steps_per_epoch = 150,
                               workers=4, shuffle = T, verbose = F)

    multilayer_prediction <- multilayer_version %>%
      predict(simulation_data[[1]])
    
    multilayer_prediction <- matrix(multilayer_prediction, ncol=401, nrow=1)
    
    write.table(multilayer_prediction, file = './code/r_code/results_simulation/results_bias_variance/results_multilayer/multilayer_bias_variance_tradeoff.csv',
                sep=";",append = TRUE,
                row.names = FALSE, col.names = FALSE)
    
    print(paste('finished iteration',i))
    rm(list=(setdiff(ls(), 'generate_data_iid_seed')))
    gc()
  }
}
