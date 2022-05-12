##------------------------------------------##
##  Author: Philipp Ratz
##  Title:  High Dimensional Regression 
##          Problem - Optimization of ANNs
##  Year:   2019
##  Tested: apple-darwin
##------------------------------------------##

rm(list=ls())
gc()
library(keras)
library(parallel)

source('~/Desktop/Thesis/code/r_code/helper_functions/data_generation.R')

function_1 <- function(x) {
  sin(x*7)^4
}
function_2 <- function(x) {
  abs(2*x)^3
}
function_3 <- function(x) {
  2*x^4
}
function_4 <- function(x) {
  -0.4*(x^2 + x^3 + 0.1*log(max(abs(x),0.5)))
}
function_5 <- function(x) {
  -4*x^3
}
function_6 <- function(x) {
  7*x^2
}
function_7 <- function(x) {
  2*log(abs(x)+0.9)^3
}
function_8 <- function(x) {
  -abs(2*x)^3
}
function_9 <- function(x) {
  -(0.9*x^2 + x^3)/(1+ sin(x)+2*x^5)
}
function_10 <- function(x) {
  -4*cos(pi*22*x)
}
function_else <- function(x, k) {
  -2*sin(x*k)
}

function_list <- list(function_1,function_2,function_3,function_4,function_5,
                      function_6,function_7,function_8,function_9,function_10,
                      function_else)

## Set up evaluation grid
relevant_elements <- c(2, 5, 15)
noise_elements <- c(1)

evaluation_setup <- expand.grid(relevant_elements,
                                noise_elements)
evaluation_setup$sums <- rowSums(evaluation_setup)
names(evaluation_setup) <- c('n_preds', 'n_noise', 'n_total')
evaluation_setup <- evaluation_setup[order(evaluation_setup$n_preds),]

# Save some compute time by re-using dataframes
evaluation_lists <- split(evaluation_setup,
                          rep(1:length(noise_elements),
                              each=length(relevant_elements),
                              length.out=dim(evaluation_setup)[1])
)

# Set up result saves
colnames_results <- matrix(c('Total_number', 'value_single'), nrow=1, ncol=2)
write.table(colnames_results, file = '~/Desktop/Thesis/code/r_code/results_simulation/results_high_dimensional/results_optimization.csv',
            sep=";",
            row.names = FALSE, col.names = FALSE)

# Run the simulations
for (i in 1:3) {
  eval_grid <- evaluation_lists[[1]]
  
  # Generate data
  sample_size = 5e2
  mean = 0 
  sd = 7
  no_irrelevant = max(eval_grid$n_noise)
  no_relevant = eval_grid$n_preds[i]
  noise_x_from = -3
  noise_x_to = 3
  x_from = 0
  x_to = 3
  train_size <- 0.8
  train_end <- ceiling(train_size*sample_size)
  
  # Simulate
  set.seed(i)
  data_geneneration <- generate_high_dim_add_iid(size = sample_size, mean = mean, sd = sd,
                                                 no_relevant = no_relevant, no_irrelevant = no_irrelevant,
                                                 noise_x_from = noise_x_from, noise_x_to = noise_x_to,
                                                 x_from = x_from, x_to = x_to,
                                                 function_list = function_list)
  
  # Shuffle and split
  set.seed(0709)
  new_order <- sample(sample_size)
  
  # Split
  df <- data_geneneration[[1]][new_order, ]
  df_train <- df[1:train_end,]
  df_test <- df[(train_end + 1):dim(df)[1],]
  
  y_train <- df_train$y
  y_test <- df_test$y 
  
  y_true <- data_geneneration[[2]][new_order]
  y_true_test <- y_true[(train_end + 1):dim(df)[1]]
  
  print(paste("Finished drawing data with", no_relevant, " relevant predictors"))
  
  ## Run models
  for (no_total_noise in unique(eval_grid$n_noise)) {
    
    use_up_to <- (no_relevant + no_total_noise)
    
    x_tmp <- as.matrix(df_train[, 2:use_up_to])
    x_test_tmp <- as.matrix(df_test[, 2:use_up_to])
    
    x_train_keras <- array_reshape(x_tmp, c(dim(x_tmp)[1], dim(x_tmp)[2]))
    x_test_keras <- array_reshape(x_test_tmp, c(dim(x_test_tmp)[1], dim(x_test_tmp)[2]))
    
    l_1_values_1 <- c(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
    evaluation_matrix <- matrix(NA, ncol=1, nrow = length(l_1_values_1))
    for(k in 1:length(l_1_values_1)) {
      
      l1_value <- l_1_values_1[k]
      
      use_session_with_seed(42)
      network <- keras_model_sequential() %>%
        layer_dense(units = 200, activation = "relu", input_shape = c((use_up_to-1)),
                    activity_regularizer = regularizer_l1(l=l1_value)) %>%
        layer_dense(units = 1, activation = 'linear')
      
      network %>% compile(
        optimizer = optimizer_adagrad(),
        loss = "mse")
      
      history <- network %>%
        fit(x_train_keras, y_train,
            epochs = 250,
            batch_size = 600,
            workers=4, shuffle = T, verbose = T)
      
      prediction_ann <- network %>%
        predict(x_test_keras)
      
      evaluation_matrix[k, 1] <- mean((prediction_ann - y_true_test)^2)

      rm(network)
      
    }
    
    set_min_values <- which.min(evaluation_matrix)
    value_single <- set_min_values

    l1_single_2 <- c(value_single*1.5, value_single*0.5)

    evaluation_matrix_1 <- matrix(NA, ncol=1, nrow = (length(l1_single_2)+1))
    evaluation_matrix_1[3, ] <- value_single
    
    for (t in 1:2) {
      l1_value_single <- l1_single_2[t]
      num_cores <- detectCores()

      use_session_with_seed(42)
      network <- keras_model_sequential() %>%
        layer_dense(units = 200, activation = "relu", input_shape = c((use_up_to-1)),
                    activity_regularizer = regularizer_l1(l=l1_value_single)) %>%
        layer_dense(units = 1, activation = 'linear')
      
      network %>% compile(
        optimizer = optimizer_adagrad(),
        loss = "mse")
      
      history <- network %>%
        fit(x_train_keras, y_train,
            epochs = 250,
            steps_per_epoch = 10,
            workers=num_cores, shuffle = T, verbose = T)
      
      prediction_ann <- network %>%
        predict(x_test_keras)
      
      evaluation_matrix_1[t, 1] <- mean((prediction_ann - y_true_test)^2)
      rm(network)
      
    }
    
    set_min_values <- apply(evaluation_matrix_1, 2, which.min)
    
    l1_single_final <- c(value_single*1.5, value_single*0.5, value_single)[set_min_values[1]]
    results_cols <- matrix(c(use_up_to, l1_single_final), nrow=1, ncol=2)
    
    write.table(results_cols, file = './code/r_code/results_simulation/results_high_dimensional/results_optimization_1.csv',
                sep=";", append=TRUE,
                row.names = FALSE, col.names = FALSE)
    
  }
}