##------------------------------------------##
##  Author: Philipp Ratz
##  Title:  High Dimensional Regression 
##          Problem - Comparison
##  Year:   2019
##  Tested: Windows >= 8
##------------------------------------------##
rm(list=ls())
gc()

library(np)
library(keras)
library(reshape2)
library(dplyr)
library(stargazer)
source('./code/r_code/helper_functions/data_generation.R')

file_ann_pred <- './code/r_code/results_simulation/results_high_dimensional/results_ann.csv'

if(file.exists(file_ann_pred)) {
  data_np <- read.csv('./code/r_code/results_simulation/results_high_dimensional/results_nonparametric.csv',
                      sep=';')
  data_np$method <- 'nonparametric'
  data_ann <- read.csv('./code/r_code/results_simulation/results_high_dimensional/results_ann.csv',
                       sep=';')
  data_ann$method <- 'ann'
  
  all_data <- rbind(data_np,
                    data_ann)
  
  summarized_data <- all_data %>%
    group_by(method, relevant_preds, noise_number) %>%
    summarise_all(funs(median)) %>%
    arrange(method)
  
  # Separate and rename
  relevant_elements <- c(2, 5, 15)
  noise_elements <- c(1, 5, 10)
  
  results_np <- data.frame(matrix(data=summarized_data$MSPE[which(summarized_data$method == 'nonparametric')],
                                  nrow = 3, ncol = 3))
  names(results_np) <- relevant_elements
  
  results_ann <- data.frame(matrix(data=summarized_data$MSPE[which(summarized_data$method == 'ann')],
                                   nrow = 3, ncol = 3))
  names(results_ann) <- relevant_elements
  all_data <- rbind(results_np,
                    results_ann)
  stargazer(all_data, summary = F, digits = 2)
} else {
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
  relevant_elements <- c(2,5,15)
  noise_elements <- c(1, 10, 20)
  
  
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
  results_np <- matrix(c("relevant_preds", "noise_number","MSPE"), nrow = 1, ncol = 3)
  write.table(results_np, file = './code/r_code/results_simulation/results_high_dimensional/results_nonparametric.csv',
              sep=";",
              row.names = FALSE, col.names = FALSE)
  
  results_ann <- matrix(c("relevant_preds", "noise_number","MSPE"), nrow = 1, ncol = 3)
  write.table(results_ann, file = './code/r_code/results_simulation/results_high_dimensional/results_ann.csv',
              sep=";",
              row.names = FALSE, col.names = FALSE)
  
  # Run the simulations
  for(k in 1:50) {
    print(paste('started iteration ', k))
    no_total_noise=1
    for (i in 1:length(evaluation_lists)) {
      # Chose grid
      eval_grid <- evaluation_lists[[i]]
      # Generate data
      sample_size = 5e2
      mean = 0 
      sd = 7
      no_irrelevant = max(eval_grid$n_noise)
      no_relevant = eval_grid$n_preds[1]
      noise_x_from = -3
      noise_x_to = 3
      x_from = 0
      x_to = 3
      train_size <- 0.8
      train_end <- ceiling(train_size*sample_size)
      
      # Simulate data
      set.seed(k)
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
        
        # Use data from optimization 
        if (no_relevant==2) {
          l1_single <- 1e-05
        } else {
          l1_single <- 1e-04
        }
        
        # Set up increasing training time, need to split because irrelevant increases fast
        
        ## Nonparametric regression
        # Get bandwidth
        bw_srt <- npcdensbw(ydat=y_train,
                            xdat=as.matrix(df_train[,2:(use_up_to+1)]),
                            cxkertype = 'gaussian',
                            bwmethod='normal-reference')
        
        bws <- bw_srt$xbw
        
        # Estimate and predict
        formula_np <- as.formula(paste('y ~ ',paste(paste("v_", c(1:use_up_to), sep="", collapse = "+"))))
        np_model <- npreg(formula=formula_np,
                          bws=bws,
                          data=df_train,
                          regtype='ll')
        
        prediction_np <- predict(np_model, newdata=df_test[,2:(use_up_to+1)])
        np_out_sample <-  sqrt(mean((prediction_np - y_true_test)^2))
        
        results_nonparametric <- matrix(c(no_relevant, no_total_noise, np_out_sample), nrow=1, ncol=3)
        
        # Save results
        write.table(results_nonparametric, file = './code/r_code/results_simulation/results_high_dimensional/results_nonparametric.csv',
                    sep=";",append = TRUE,
                    row.names = FALSE, col.names = FALSE)
        
        print(paste("Finished np regression with", no_total_noise, "noise and",no_relevant, "relevant preds"))
        rm(np_model, prediction_np, np_out_sample, results_nonparametric, bws, bw_srt)
        gc()
        
        # ## ANN
        # # Keras df needs some reshaping
        
        if(no_relevant == 2) {
          training_epochs <- 200
        }
        if(no_relevant == 5) {
          training_epochs <- 300
        }
        if(no_relevant == 15) {
          training_epochs <- 500 
        }
        
        x_tmp <- as.matrix(df_train[, 2:use_up_to])
        x_test_tmp <- as.matrix(df_test[, 2:use_up_to])
        
        x_train_keras <- array_reshape(x_tmp, c(dim(x_tmp)[1], dim(x_tmp)[2]))
        x_test_keras <- array_reshape(x_test_tmp, c(dim(x_test_tmp)[1], dim(x_test_tmp)[2]))
        
        network <- keras_model_sequential() %>%
          layer_dense(units = 200, activation = "relu", input_shape = c((use_up_to-1)),
                      activity_regularizer = regularizer_l1(l=l1_single)) %>%
          layer_dense(units = 1, activation = 'linear')
        
        network %>% compile(
          optimizer = optimizer_adagrad(),
          loss = "mse")
        
        if (no_relevant ==2 & no_total_noise ==1) {
          network %>% compile(
            optimizer = optimizer_adagrad(lr=0.02),
            loss = "mse")
        }
        
        history <- network %>%
          fit(x_train_keras, y_train,
              steps_per_epoch = 10,
              epochs = training_epochs,
              workers=8, shuffle = T, verbose = F)
        print(training_epochs)
        
        prediction_ann <- network %>%
          predict(x_test_keras)
        
        ann_out_sample <- sqrt(mean((prediction_ann- y_true_test)^2))
        results_ann <- matrix(c(no_relevant, no_total_noise, ann_out_sample), nrow=1, ncol=3)
        
        # Save results
        if(k == 1 & no_relevant == 2) {
          save(history,'./code/r_code/results_simulation/results_high_dimensional/history_2')
        }
        if(k == 1 & no_relevant == 5) {
          save(history,'./code/r_code/results_simulation/results_high_dimensional/history_5')
        }
        if(k == 1 & no_relevant == 15) {
          save(history,'./code/r_code/results_simulation/results_high_dimensional/history_15')
        }
        
        write.table(results_ann, file = './code/r_code/results_simulation/results_high_dimensional/results_ann.csv',
                    sep=";",append = TRUE,
                    row.names = FALSE, col.names = FALSE)
        
        print(paste("Finished ANN with", no_total_noise, "noise and",no_relevant, "relevant preds"))
        rm(network, prediction_ann, ann_out_sample, results_ann)
        gc()
      }
    }
  }
}

## Loss plots (need to be extracted separately) - ugly code but hey - it works
load('./code/r_code/results_simulation/results_high_dimensional/history_2')
history_2 <- history
rm(history)
load('./code/r_code/results_simulation/results_high_dimensional/history_5')
history_5 <- history
rm(history)
load('./code/r_code/results_simulation/results_high_dimensional/history_15')
history_15 <- history
rm(history)

par(mfrow=c(1,3))
png('./tex_files/thesis/Figures/r_figures/history_high_dim_sim.png', 
    width = 629, height = 444)
plot(history_2$metrics$loss, ylim=c(100, 10000), ylab="", xlab='Epochs', main='2 relevant predictors')
plot(history_5$metrics$loss, ylim=c(100, 10000), ylab="", xlab='Epochs', main='5 relevant predictors')
plot(history_15$metrics$loss, ylim=c(100, 10000), ylab="", xlab='Epochs', main='15 relevant predictors')
dev.off()