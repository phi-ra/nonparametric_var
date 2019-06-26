##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Inference and prediction accuracy
##         for time series (stochastic)
##         Comparison linear-ANN-semiparametric
## Tested: Apple-Darwin/Linux-Mint
##----------------------------------------------##

rm(list=ls())
gc()
library(keras)
library(nlme)
library(stargazer)
library(dplyr)
library(reshape2)

#### Simulation ####

mse_file_string <- './code/r_code/results_simulation/results_stochastic_timeseries/results_numeric/params_linear.csv'

if(file.exists(mse_file_string)) {
  
  parameters_linear <- read.csv('./code/r_code/results_simulation/results_stochastic_timeseries/results_numeric/params_linear.csv',
                                sep=';')
  
  parameters_twostep <- read.csv('./code/r_code/results_simulation/results_stochastic_timeseries/results_numeric/parameters_twostep.csv',
                                 sep=';')
  
  mse_all <- read.csv('./code/r_code/results_simulation/results_stochastic_timeseries/results_numeric/all_mse.csv',
                      sep=';')
} else {
  colnames_mse <- matrix(c('iteration', 'linear', 'hybrid', 'ann_pure', 'incorrect_spec', 'correct_spec'),
                         nrow = 1)
  
  write.table(colnames_mse, file = './code/r_code/results_simulation/results_stochastic_timeseries/results_numeric/all_mse.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  colnames_linear <-  matrix(c('iteration',
                               'intercept', 'param_x1','param_x2', 'param_u1','param_u2',
                               'sd_intercept', 'sd_param_x1','sd_param_x2', 'sd_param_u1','sd_param_u2'),
                             nrow=1)
  write.table(colnames_linear, file = './code/r_code/results_simulation/results_stochastic_timeseries/results_numeric/params_linear.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  colnames_hybrid_twostep <- matrix(c('iteration',
                                      'intercept_unfiltered', 'param_1_unfiltered', ' param_2_unfiltered',
                                      'param_1_filtered', ' param_2_filtered',
                                      'sd_1_filtered', ' sd_2_filtered'),
                                    nrow=1)
  
  write.table(colnames_hybrid_twostep, file = './code/r_code/results_simulation/results_stochastic_timeseries/results_numeric/parameters_twostep.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  size=1000/0.8
  train=0.8*size
  test=train+1
  lags=2
  num_iterations=50
  for (j in 1:num_iterations){
    ## Generate and lag time series
    set.seed(j)
    err <- rnorm(size, sd=0.5)
    del <- runif(size, -0.5,0.5)
    eps <- runif(size, -0.5,0.5)
    
    x <- numeric(size)
    y <- numeric(size)
    u <- numeric(size)
    
    for (i in 3:size) {
      x[i] <- 0.8*sin(2*pi*x[i-1]) - 0.2*cos(2*pi*x[i-2]) + del[i]
    }
    
    for (i in 3:size) {
      u[i] <- 0.55*u[i-1] - 0.12*u[i-2] + del[i]
    }
    
    for (i in 3:size) {
      y[i] <- 0.47*u[i-1] - 0.45*u[i-2] - (((x[i-1] + x[i-2])/(1 + (x[i-1])^2 + (x[i-2])^2)))^2 + err[i] 
    }
    
    source('./code/r_code/helper_functions/data_generation.R')
    x_lagged <- get_lagged_data(lags, x[3:length(x)])[[1]]
    u_lagged <- get_lagged_data(lags, u[3:length(u)])[[1]]
    
    input_df_tmp <- data.frame(y[(lags+3):size], x_lagged, u_lagged)
    names(input_df_tmp) <- c('y', paste('x_', c(1:lags), sep = ""), paste('u_', c(1:lags), sep = ""))
    
    ## Linear model
    linmod <- lm(y ~ ., data=input_df_tmp[1:train,])
    param_linmod <- unname(coef(summary(linmod))[, 'Estimate'])
    se_linmod <- unname(coef(summary(linmod))[,'Std. Error'])
    
    pred_linmod <- predict(linmod, input_df_tmp[test:dim(input_df_tmp)[1], c(2:5)])
    mse_linmod <- mean((input_df_tmp[test:dim(input_df_tmp)[1],1][1:246] - pred_linmod[1:246])^2)
    
    coefs_sd_linmod <- c(param_linmod, se_linmod)
    results_linmod <- matrix(c(j, coefs_sd_linmod), nrow=1)
    
    write.table(results_linmod, file = './code/r_code/results_simulation/results_stochastic_timeseries/results_numeric/params_linear.csv',
                append = TRUE,
                sep=";", row.names = FALSE, col.names = FALSE)
    
    ## Keras Models
    
    # Data transformations
    x_train_nl <- array_reshape(as.matrix(input_df_tmp[1:train,2:(lags+1)]), c(train, lags))
    x_train_l <- array_reshape(as.matrix(input_df_tmp[1:train,(lags+2):(dim(input_df_tmp)[2])]), c(train, lags))
    x_train_both <- array_reshape(as.matrix(input_df_tmp[1:train,2:(dim(input_df_tmp)[2])]), c(train, (2*lags)))
    y_train <- array_reshape(as.matrix(input_df_tmp[1:train, 1]), c(train,1))
    
    x_test_nl <- array_reshape(as.matrix(input_df_tmp[test:size,2:(lags+1)]), c((size-train), lags))
    x_test_l <- array_reshape(as.matrix(input_df_tmp[test:size,(lags+2):(dim(input_df_tmp)[2])]), c((size-train),lags))
    x_test_both <- array_reshape(as.matrix(input_df_tmp[test:size,2:(dim(input_df_tmp)[2])]), c((size-train), (2*lags)))
    y_test <- array_reshape(as.matrix(input_df_tmp[test:size, 1]), c((size-train),1))
    
    # Set up Hybrid model
    use_session_with_seed(j)
    nonlinear_input <- layer_input(shape = c(2), name='nonlin_input')
    nonlinear <- nonlinear_input %>%
      layer_dense(units =70, activation = 'relu' , activity_regularizer = regularizer_l1(l=0.00005)) %>%
      layer_dense(units = 5 , activation = 'relu')
    
    # Add linear layer and combine
    linear_input <- layer_input(shape = c(2), name = 'linear_input')
    both_layers <- layer_concatenate(list(nonlinear, linear_input))
    output <- both_layers %>%
      layer_dense(units=1, activation = 'linear', use_bias = T)
    model_hybrid <- keras_model(inputs = list(nonlinear_input, linear_input),
                                outputs = output)
    
    # Add estimation
    model_hybrid %>% compile(
      optimizer = optimizer_adagrad(),
      loss = 'mse'
    )
    
    # Fit Hybrid
    model_hybrid %>%
      fit(x= list(nonlin_input=x_train_nl, linear_input=x_train_l),
          y=y_train, steps_per_epoch = 200,
          epochs = 50, verbose = T)
    
    prediction_hybrid <- model_hybrid %>%
      predict(list(x_test_nl, x_test_l))
    
    mse_hybrid <- mean((y_test[1:246] - prediction_hybrid[1:246])^2)
    
    ## Non specific setup
    
    use_session_with_seed(j)
    nonlinear_input <- layer_input(shape = c(4), name='nonlin_input')
    nonlinear <- nonlinear_input %>%
      layer_dense(units =70, activation = 'relu' , activity_regularizer = regularizer_l1(l=0.0005)) %>%
      layer_dense(units = 5 , activation = 'relu')
    
    # Add linear layer and combine
    linear_input <- layer_input(shape = c(4), name = 'linear_input')
    both_layers <- layer_concatenate(list(nonlinear, linear_input))
    output <- both_layers %>%
      layer_dense(units=1, activation = 'linear', use_bias = T)
    model_hybrid <- keras_model(inputs = list(nonlinear_input, linear_input),
                                outputs = output)
    
    # Add estimation
    model_hybrid %>% compile(
      optimizer = optimizer_adagrad(),
      loss = 'mse'
    )
    
    # Fit Hybrid
    model_hybrid %>%
      fit(x= list(nonlin_input=x_train_both, linear_input=x_train_both),
          y=y_train, steps_per_epoch = 200,
          epochs = 50, verbose = T)
    
    prediction_hybrid <- model_hybrid %>%
      predict(list(x_test_both, x_test_both))
    
    mse_noncorrect <- mean((y_test[1:246] - prediction_hybrid[1:246])^2)
    
    ## Two step inference
    
    use_session_with_seed(j)
    nonparametric <- keras_model_sequential() %>%
      layer_dense(units = 50, activation = 'relu', input_shape = c(2), activity_regularizer = regularizer_l1(l=0.000005)) %>%
      layer_dense(units = 5 , activation = 'relu') %>%
      layer_dense(units=1, activation='linear')
    
    nonparametric %>% compile(
      optimizer = optimizer_adagrad(),
      loss = 'mse'
    )
    
    nonparametric %>%
      fit(x=x_train_nl,
          y=y_train, steps_per_epoch = 100,
          epochs = 100, verbose = F)
    
    if( unname(Sys.info()['sysname'])=='Darwin') {
      layer_name <- 'dense_2'
    } else {
      layer_name <- 'dense_1'
    }
    
    last_layer <- keras_model(inputs = nonparametric$input,
                              outputs = get_layer(nonparametric, layer_name)$output)
    
    df_layer <- data.frame(predict(last_layer, x_train_nl))
    df_final <- data.frame(cbind(y_train, x_train_l, df_layer))
    
    df_layer_t <- data.frame(predict(last_layer, x_test_nl))
    df_final_t <- data.frame(cbind(y_test, x_test_l, df_layer_t))
    
    lm_both <- lm(y_train ~ ., data=df_final)
    coefs_twostep_nonfiltered <- unname(coef(summary(lm_both))[,'Estimate'])[1:3]
    
    # Filter
    z_mat <- as.matrix(df_final[,c(4:8)])
    x_mat <- as.matrix(df_final[,c(2:3)])
    y_mat <- as.matrix(df_final[,1])
    
    y_on_z <- lm(y_mat ~ 0 + z_mat)
    y_wiggle <- (y_mat- predict(y_on_z))
    x_on_z <- lm(x_mat ~ 0 + z_mat)
    x_wiggle <- (x_mat - predict(x_on_z))
    
    final_reg <- lm(y_wiggle ~ 0 + x_wiggle)
    filtered_params <- unname(coef(summary(final_reg))[, 'Estimate'])
    filtered_sd <- unname(coef(summary(final_reg))[, 'Std. Error'])
    
    parameter_sd_twostep <- c(coefs_twostep_nonfiltered, filtered_params, filtered_sd)
    results_twostep <- matrix(c(j, parameter_sd_twostep),
                              nrow=1)
    
    write.table(results_twostep, file = './code/r_code/results_simulation/results_stochastic_timeseries/results_numeric/parameters_twostep.csv',
                sep=";",append = TRUE,
                row.names = FALSE, col.names = FALSE)
    
    ## Pure ANN
    use_session_with_seed(j)
    ann_pure <- keras_model_sequential() %>%
      layer_dense(units = 70, activation = 'relu', activity_regularizer = regularizer_l1(l=0.000005)) %>%
      layer_dense(units = 5 , activation = 'relu') %>%
      layer_dense(units=1, activation='linear')
    
    ann_pure %>% compile(
      optimizer = optimizer_adagrad(),
      loss = 'mse'
    )
    
    # fit
    ann_pure %>%
      fit(x= x_train_both,
          y=y_train, steps_per_epoch = 200,
          epochs = 50, verbose = T)
    
    prediction_ann <- ann_pure %>%
      predict(x_test_both)
    
    mse_ann <- mean((y_test[1:246] - prediction_ann[1:246])^2)
    
    # Correctly specified model
    nonlin_x <- ((x_lagged[,1] + x_lagged[,2])/(1 + x_lagged[,1]^2 + x_lagged[,2]^2))^2
    df_ideal <- data.frame(y[(lags+3):size], u_lagged, nonlin_x)
    names(df_ideal) <- c('y', 'u1', 'u2', 'x')
    
    linmod_ideal <- lm(y ~ ., data=df_ideal[1:train, ])
    pred_ideal <- predict(linmod_ideal, df_ideal[test:dim(df_ideal)[1], c(2:4)])
    mse_ideal <- mean((df_ideal[test:dim(df_ideal)[1],1] - pred_ideal)^2)
    
    all_mse <- c(mse_linmod, mse_hybrid, mse_ann, mse_noncorrect, mse_ideal)
    
    results_mse <- matrix(c(j, all_mse),
                          nrow=1)
    
    write.table(results_mse, file = './code/r_code/results_simulation/results_stochastic_timeseries/results_numeric/all_mse.csv',
                sep=";",append = TRUE,
                row.names = FALSE, col.names = FALSE)
    print(paste('finished iteration', j))
    rm(list=setdiff(ls(), c("size", 'train', 'test', 'lags', 'num_iterations')))
    gc()
  }
}

#### Visualisation #### 

## Parameters and tables 
png('./tex_files/thesis/Figures/r_figures/figure_parametric_estimate.png', 
    width = 629, height = 444)
plot(density(parameters_linear$param_u1), col='black', ylim=c(0,10), ylab = "", xlab = 'Estimate',
     main="", lwd='2')
lines(density(parameters_twostep$param_1_filtered), lwd=2, col='red')
abline(v=0.47, lty='dotted', lwd=2, col='lightgrey')
legend('topright',
       legend = c('OLS', 'SANN', 'Truth'),
       col=c('black', 'red', 'lightgrey'),
       lty=c(1,1,2),
       lwd=1.5)
dev.off()

png('./tex_files/thesis/Figures/r_figures/figure_standard_errors.png', 
    width = 629, height = 444)
plot(density(parameters_linear$sd_param_u1), col='black', ylab = "",
     xlim=c(0.04,0.067), xlab = 'Standard Errors',
     ylim=c(0,250),
     main="", lwd=2)
lines(density(parameters_twostep$sd_1_filtered), lwd=2, col='red')
abline(v=mean(parameters_linear$sd_param_u1), lty='dotted', lwd=2, col='black')
abline(v=mean(parameters_twostep$sd_1_filtered), lty='dotted', lwd=2, col='red')
legend('topright',
       legend = c('OLS', 'SANN'),
       col=c('black', 'red'),
       lty=c(1,1),
       lwd=1.5)
dev.off()

## Performance Table
mean_linear <- round(mean(sqrt(mse_all$linear)),4)
mean_semi <- round(mean(sqrt(mse_all$hybrid)),4)
mean_ann <- round(mean(sqrt(mse_all$ann_pure)),4)
mean_model2 <- round(mean(sqrt(mse_all$incorrect_spec)),4)
mean_correct <- round(mean(sqrt(mse_all$correct_spec)), 4)

result_table <- matrix(c( mean_correct ,mean_linear, mean_ann, mean_semi, 
                         NA,NA, NA, mean_model2),
                       ncol=2, nrow = 4, byrow = F)

colnames(result_table) <- c('Model 1', 'Model 2')
rownames(result_table) <- c('Correct', 'Linear', 'ANN', 'SANN')
stargazer(result_table, digits = 4)
mean_model2
