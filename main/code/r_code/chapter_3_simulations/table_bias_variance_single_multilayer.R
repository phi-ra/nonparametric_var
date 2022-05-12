##------------------------------------------##
##  Author: Philipp Ratz
##  Title:  Calculations table for Bias-
##          Variance calculations Single
##          and Multilayer 
##  Year:   2019
##  Tested: apple-darwin
##------------------------------------------##
rm(list=ls())
gc()

# Packages
library(dplyr)
source('./code/r_code/helper_functions/data_generation.R')

# Load data from monte carlo simulations
file_mutlivariateann <- './code/r_code/results_simulation/results_bias_variance/results_multilayer/multilayer_bias_variance_tradeoff.csv'
if(file.exists(file_mutlivariateann)) {
  single <- read.csv(file='./code/r_code/results_simulation/results_bias_variance/results_singlelayer/singlelayer_bias_variance_tradeoff.csv',
                     sep=';', header = T)
  multi <- read.csv(file='./code/r_code/results_simulation/results_bias_variance/results_multilayer/multilayer_bias_variance_tradeoff.csv',
                    sep=';', header = T)
  
  colnames(multi) <- paste('x_', c(1:401), sep='')
  colnames(single) <- paste('x_', c(1:401), sep='')
  
  multi$type <- 'multi'
  single$type <- 'single'
  multi <- as.data.frame(multi)
  single <- as.data.frame(single)
  
  df <- rbind(multi, single)
  set.seed(42)
  truth  <- generate_data_iid(size = 400, mean=0, sd=441,
                              x_from=-10, x_to = 10, break_at = 1)[[3]]
  
  averaged_prediction <- df %>%
    group_by(type) %>%
    summarise_all(funs(mean))
  
  df_bias_tmp <- sweep(as.matrix(averaged_prediction[,2:402]),2,truth)
  df_bias <- data.frame(df_bias_tmp^2)
  avg_bias <- rowMeans(df_bias)
  
  ##
  variance_multi <- sweep(as.matrix(df[which(df$type == 'multi'),1:401]), 2, as.matrix(averaged_prediction[1,2:402]))
  variance_single <- sweep(as.matrix(df[which(df$type == 'single'),1:401]), 2, as.matrix(averaged_prediction[2,2:402]))
  
  var_m <- data.frame(variance_multi^2)
  var_s <- data.frame(variance_single^2)
  
  var_m$type <- 'multi'
  var_s$type <- 'single'
  
  var_df <- rbind(var_m, var_s)
  
  variance_final <- var_df %>%
    group_by(type) %>%
    summarise_all(funs(mean))
  
  
  ## MSE 
  mse_all <- sweep(as.matrix(df[,1:401]),2,truth)
  mse_all <- as.data.frame(mse_all^2)
  
  mse_all$type <- df$type
  
  mse_multi <- mse_all %>%
    group_by(type) %>%
    summarise_all(funs(mean))
  ## Summarize all in table
  mse <- rowMeans(mse_multi[,2:402])
  var <- rowMeans(variance_final[,2:402])
  bias <- rowMeans(df_bias)
  
  all <- data.frame(cbind(bias,
                          var,
                          mse))
  
  colnames(all) <- c('Bias', 'Var', 'MSE')
  rownames(all) <- c('Multilayer', 'Singlelayer')
  stargazer(all, summary = FALSE, digits = 0)
} else {
  print('You need to run the simulations first - (./code/r_code/chapter_2_simulations/bias_variance_decomposition/)')
}

#### History plots ####
history_single <- read.csv(file = './code/r_code/results_simulation/results_bias_variance/results_singlelayer/history_singlelayer_network.csv',
                           sep=';', header = F)
history_multi <- read.csv(file = './code/r_code/results_simulation/results_bias_variance/results_multilayer/history_multilayer_network.csv',
                          sep=';', header=F)
history_single <- t(history_single)
history_multi <- t(history_multi)

par(mfrow=c(1,2))
png('./tex_files/thesis/Figures/r_figures/loss_epochs_single_vs_multilayer.png', 
    width = 932, height = 455)
plot(history_single[3:400], ylim=c(1e05, 6e05), ylab="", xlab='Epochs', main='Singlelayer')
plot(history_multi[3:400], ylim=c(1e05, 6e05), ylab="", xlab='Epochs', main='Multilayer')
dev.off()
