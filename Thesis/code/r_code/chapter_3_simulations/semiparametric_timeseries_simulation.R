##----------------------------------------------##
## Author: Philipp Ratz
##         University of Bonn
## Year:   2019
## Title:  Inference and prediction accuracy
##         for time series
##         Comparison Linear-ANN-SANN
## Tested: Apple-Darwin
##----------------------------------------------##
rm(list=ls())
gc()

## Packages and set up for data
library(nlme)
library(keras)
library(PLRModels)
library(sfsmisc)
library(dplyr)

file_nonparametric_component <- './code/r_code/results_simulation/results_semiparametric_model/results_numeric/nonparametric_estimate.csv'

if(file.exists(file_nonparametric_component)) {
  mse_dataframe <- read.csv('./code/r_code/results_simulation/results_semiparametric_model/results_numeric/all_mse.csv',
                            sep=';')
  
  parameters_linear <- read.csv('./code/r_code/results_simulation/results_semiparametric_model/results_numeric/params_linear.csv',
                                sep=';')
  
  parameters_twostep <- read.csv('./code/r_code/results_simulation/results_semiparametric_model/results_numeric/parameters_twostep.csv',
                                 sep=';')
  
  estimation_nonparametric_plm <- read.csv('./code/r_code/results_simulation/results_semiparametric_model/results_numeric/nonparametric_estimate_plm.csv',
                                           sep=';')
  
  estimation_nonparametric_ann <- read.csv('./code/r_code/results_simulation/results_semiparametric_model/results_numeric/nonparametric_estimate.csv',
                                           sep=';')
  
  ideal_mse <- read.csv('./code/r_code/results_simulation/results_semiparametric_model/results_numeric/mse_ideal.csv',
                        sep=';')
} else {
  results_noparam <- matrix(c(1:1000), nrow=1, ncol=1000)
  write.table(results_noparam, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/nonparametric_estimate.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  
  colnames_mse <- matrix(c('iteration', 'linear', 'hybrid', 'twostep_hybrid', 'ann_pure'),
                         nrow = 1)
  
  write.table(colnames_mse, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/all_mse.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  colnames_linear <-  matrix(c('iteration',
                               'intercept', 'param_x1','param_x2', 'param_u1','param_u2',
                               'sd_intercept', 'sd_param_x1','sd_param_x2', 'sd_param_u1','sd_param_u2'),
                             nrow=1)
  write.table(colnames_linear, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/params_linear.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  colnames_hybrid_twostep <- matrix(c('iteration',
                                      'intercept_unfiltered', 'param_1_unfiltered', ' param_2_unfiltered',
                                      'param_1_filtered', ' param_2_filtered',
                                      'sd_1_filtered', ' sd_2_filtered'),
                                    nrow=1)
  
  write.table(colnames_hybrid_twostep, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/parameters_twostep.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  colnames_param_plm <- matrix(c('beta_1', 'beta_2'), nrow=1)
  write.table(colnames_param_plm, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/parametric_estimate_plm.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  nonparametrics_plm <- matrix(c(1:1000), nrow=1)
  write.table(nonparametrics_plm, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/nonparametric_estimate_plm.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  mse_ideal <- matrix(c('MSE'), nrow=1, ncol=1)
  write.table(mse_ideal, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/mse_ideal.csv',
              sep=";", row.names = FALSE, col.names = FALSE)
  
  #### Simulation ####
  size=1000/0.8
  train=0.8*size
  test=train+1
  lags=2
  num_iterations=50
  
  for (j in 1:num_iterations){
    
    # Fix x_3 because we need it for the IMSE
    set.seed(27)
    x_3 <- runif(size, -2,2)
    # Generate data
    set.seed(j)
    x_2 <- rt(n = size, df=4)
    x_1 <- 0.5*x_3 + rnorm(size)
    u <- rnorm(size, 0, sd=sqrt(0.1))
    h1 <- 0.3*exp(-4*(x_3 + 1)^2) + 0.7*exp(-16*(x_3 - 1)^2)
    y <- 2*x_1 + x_2 + 0.3*exp(-4*(x_3 + 1)^2) + 0.7*exp(-16*(x_3 - 1)^2) + u
    
    # Selection of the nonparametric components
    input_df_tmp <- data.frame(y, x_1,x_2,x_3)
    
    ## Ideal model 
    ideal_df <- data.frame(y, x_1,x_2,h1)
    linmod_ideal <- lm(y ~ ., data=ideal_df[1:train, ])
    
    pred_ideal <- predict(linmod_ideal, ideal_df[test:dim(ideal_df)[1], c(2:4)])
    mse_ideal <- mean((ideal_df[test:dim(ideal_df)[1],1] - pred_ideal)^2)
    
    write.table(mse_ideal, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/mse_ideal.csv',
                append = TRUE,
                sep=";", row.names = FALSE, col.names = FALSE)
    
    ## Linear model
    linmod <- lm(y ~ ., data=input_df_tmp[1:train,])
    param_linmod <- unname(coef(summary(linmod))[, 'Estimate'])
    se_linmod <- unname(coef(summary(linmod))[,'Std. Error'])
    
    pred_linmod <- predict(linmod, input_df_tmp[test:dim(input_df_tmp)[1], c(2:4)])
    mse_linmod <- mean((input_df_tmp[test:dim(input_df_tmp)[1],1][1:246] - pred_linmod[1:246])^2)
    
    coefs_sd_linmod <- c(param_linmod, se_linmod)
    results_linmod <- matrix(c(j, coefs_sd_linmod), nrow=1)
    
    write.table(results_linmod, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/params_linear.csv',
                append = TRUE,
                sep=";", row.names = FALSE, col.names = FALSE)
    
    
    ## Keras
    
    # Data transformations
    x_train_nl <- array_reshape(as.matrix(input_df_tmp[1:train,4]), c(train, 1))
    x_train_l <- array_reshape(as.matrix(input_df_tmp[1:train,2:3]), c(train, 2))
    x_train_both <- array_reshape(as.matrix(input_df_tmp[1:train,2:(dim(input_df_tmp)[2])]), c(train, (3)))
    y_train <- array_reshape(as.matrix(input_df_tmp[1:train, 1]), c(train,1))
    
    x_test_nl <- array_reshape(as.matrix(input_df_tmp[test:size,4]), c((size-train), 1))
    x_test_l <- array_reshape(as.matrix(input_df_tmp[test:size,2:3]), c((size-train),lags))
    x_test_both <- array_reshape(as.matrix(input_df_tmp[test:size,2:(dim(input_df_tmp)[2])]), c((size-train), (3)))
    y_test <- array_reshape(as.matrix(input_df_tmp[test:size, 1]), c((size-train),1))
    
    # Set up Hybrid model
    use_session_with_seed(42)
    nonlinear_input <- layer_input(shape = c(1), name='nonlin_input')
    nonlinear <- nonlinear_input %>%
      layer_dense(units = 150, activation = 'relu', activity_regularizer = regularizer_l1(l=0.000001)) %>%
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
      optimizer = optimizer_adagrad(), #optimizer_adagrad(lr=0.08),
      loss = 'mse'
    )
    
    # Fit Hybrid
    model_hybrid %>%
      fit(x= list(nonlin_input=x_train_nl, linear_input=x_train_l),
          y=y_train, steps_per_epoch = 200,
          epochs = 100, verbose = F)
    
    prediction_hybrid <- model_hybrid %>%
      predict(list(x_test_nl, x_test_l))
    mse_hybrid <- mean((y_test[1:246] - prediction_hybrid[1:246])^2)
    
    pred_insample <- model_hybrid %>%
      predict(list(x_train_nl, x_train_l))
    
    weighted_mat <- as.matrix(x_train_l) %*% as.matrix(c(get_weights(model_hybrid)[[5]][6:7]))
    leftover <- pred_insample - weighted_mat
    leftover_format <- matrix(leftover, nrow=1, ncol=1000)
    
    write.table(leftover_format, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/nonparametric_estimate.csv',
                append = TRUE,
                sep=";", row.names = FALSE, col.names = FALSE)
    
    ## Two step inference
    use_session_with_seed(42)
    nonparametric <- keras_model_sequential() %>%
      layer_dense(units = 120, activation = 'relu', input_shape = c(1), activity_regularizer = regularizer_l1(l=0.000001)) %>%
      layer_dense(units = 5 , activation = 'relu') %>%
      layer_dense(units=1, activation='linear')
    
    nonparametric %>% compile(
      optimizer = optimizer_adagrad(),
      loss = 'mse'
    )
    
    nonparametric %>%
      fit(x=x_train_nl,
          y=y_train, steps_per_epoch = 200,
          epochs = 100, verbose = F)
    
    np_pred <- nonparametric %>%
      predict(x_train_nl)
    
    layer_name <- 'dense_2'
    last_layer <- keras_model(inputs = nonparametric$input,
                              outputs = get_layer(nonparametric, layer_name)$output)
    
    df_layer <- data.frame(predict(last_layer, x_train_nl))
    df_final <- data.frame(cbind(y_train, x_train_l, df_layer))
    
    df_layer_t <- data.frame(predict(last_layer, x_test_nl))
    df_final_t <- data.frame(cbind(y_test, x_test_l, df_layer_t))
    
    lm_both <- lm(y_train ~ ., data=df_final)
    coefs_twostep_nonfiltered <- unname(coef(summary(lm_both))[,'Estimate'])[1:3]
    
    predict_lm <- predict(lm_both, df_final_t[,c(2:8)])
    mse_twosetp <- mean((y_test[1:246] - predict_lm[1:246])^2)
    
    # Filter
    z_mat <- as.matrix(df_final[,c(4:8)])
    x_mat <- as.matrix(df_final[,c(2:3)])
    y_mat <- as.matrix(df_final[,1])
    
    y_on_z <- lm(y_mat ~ z_mat)
    y_wiggle <- (y_mat- predict(y_on_z))
    x_on_z <- lm(x_mat ~ z_mat)
    x_wiggle <- (x_mat - predict(x_on_z))
    
    final_reg <- lm(y_wiggle ~ 0 + x_wiggle)
    
    filtered_params <- unname(coef(summary(final_reg))[, 'Estimate'])
    filtered_sd <- unname(coef(summary(final_reg))[, 'Std. Error'])
    
    parameter_sd_twostep <- c(coefs_twostep_nonfiltered, filtered_params, filtered_sd)
    results_twostep <- matrix(c(j, parameter_sd_twostep),
                              nrow=1)
    
    write.table(results_twostep, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/parameters_twostep.csv',
                sep=";",append = TRUE,
                row.names = FALSE, col.names = FALSE)
    
    ## Pure ANN
    use_session_with_seed(42)
    ann_pure <- keras_model_sequential() %>%
      layer_dense(units = 150, activation = 'relu', activity_regularizer = regularizer_l1(l=0.000001)) %>%
      layer_dense(units = 5 , activation = 'relu') %>%
      layer_dense(units=1, activation='linear')
    
    ann_pure %>% compile(
      optimizer = optimizer_adagrad(),
      loss = 'mse'
    )
    
    # fit
    history <- ann_pure %>%
      fit(x= x_train_both,
          y=y_train, steps_per_epoch = 200,
          epochs = 100, verbose = F)
    
    prediction_ann <- ann_pure %>%
      predict(x_test_both)
    
    mse_ann <- mean((y_test[1:246] - prediction_ann[1:246])^2)
    
    all_mse <- c(mse_linmod, mse_hybrid, mse_twosetp, mse_ann)
    
    results_mse <- matrix(c(j, all_mse),
                          nrow=1)
    
    write.table(results_mse, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/all_mse.csv',
                sep=";",append = TRUE,
                row.names = FALSE, col.names = FALSE)
    
    ## Kernel PLS
    # Modify to avoid bug
    plr_matrix <- as.matrix(input_df_tmp[1:train, ])
    plr_matrix[,4] <- plr_matrix[,4] + 2.000000000002
    
    # Fit model
    plm_model <- plrm.est(plr_matrix)
    nonparametric_estimate <- plm_model$m.t
    parametric_estimate <- plm_model$beta
    
    nonparametric_plm <- matrix(c(nonparametric_estimate), nrow=1)
    parameters_plm <- matrix(c(parametric_estimate), nrow=1)
    
    write.table(parameters_plm, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/parametric_estimate_plm.csv',
                sep=";",append = TRUE,
                row.names = FALSE, col.names = FALSE)
    
    write.table(nonparametric_plm, file = './code/r_code/results_simulation/results_semiparametric_model/results_numeric/nonparametric_estimate_plm.csv',
                sep=";",append = TRUE,
                row.names = FALSE, col.names = FALSE)
    
    print(paste('finished iteration', j))
    rm(list=setdiff(ls(), c("size", 'train', 'test', 'lags', 'num_iterations')))
    gc()
  }
}

file_visualisation <- './code/r_code/results_simulation/results_semiparametric_model/visualisation_approximation/grid_x'
if(file.exists(file_visualisation)) {
  load(file = './code/r_code/results_simulation/results_semiparametric_model/visualisation_approximation/grid_x')
  load(file = './code/r_code/results_simulation/results_semiparametric_model/visualisation_approximation/grid_h1')
  load(file = './code/r_code/results_simulation/results_semiparametric_model/visualisation_approximation/nonlinear_comp_ann')
  load(file = './code/r_code/results_simulation/results_semiparametric_model/visualisation_approximation/nonparametric_fit_plm')
} else {
  
  # Redraw data
  size=1000/0.8
  train=0.8*size
  test=train+1
  lags=2
  
  set.seed(27)
  x_3 <- runif(size, -2,2)
  x_2 <- rt(n = size, df=4)
  x_1 <- 0.5*x_3 + rnorm(size)
  u <- rnorm(size, 0, sd=sqrt(0.1))
  h1 <- 0.3*exp(-4*(x_3 + 1)^2) + 0.7*exp(-16*(x_3 - 1)^2)
  y <- 2*x_1 + x_2 + 0.3*exp(-4*(x_3 + 1)^2) + 0.7*exp(-16*(x_3 - 1)^2) + u
  
  input_df_tmp <- data.frame(y, x_1,x_2,x_3)
  
  x_train_nl <- array_reshape(as.matrix(input_df_tmp[1:train,4]), c(train, 1))
  x_train_l <- array_reshape(as.matrix(input_df_tmp[1:train,2:3]), c(train, 2))
  x_train_both <- array_reshape(as.matrix(input_df_tmp[1:train,2:(dim(input_df_tmp)[2])]), c(train, (3)))
  y_train <- array_reshape(as.matrix(input_df_tmp[1:train, 1]), c(train,1))
  
  x_test_nl <- array_reshape(as.matrix(input_df_tmp[test:size,4]), c((size-train), 1))
  x_test_l <- array_reshape(as.matrix(input_df_tmp[test:size,2:3]), c((size-train),lags))
  x_test_both <- array_reshape(as.matrix(input_df_tmp[test:size,2:(dim(input_df_tmp)[2])]), c((size-train), (3)))
  y_test <- array_reshape(as.matrix(input_df_tmp[test:size, 1]), c((size-train),1))
  
  use_session_with_seed(42)
  nonlinear_input <- layer_input(shape = c(1), name='nonlin_input')
  nonlinear <- nonlinear_input %>%
    layer_dense(units = 70, activation = 'relu', activity_regularizer = regularizer_l1(l=0.00001)) %>%
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
    optimizer = optimizer_adam(), 
    loss = 'mse'
  )
  
  # Fit Hybrid
  model_hybrid %>%
    fit(x= list(nonlin_input=x_train_nl, linear_input=x_train_l),
        y=y_train, steps_per_epoch = 200,
        epochs = 30, verbose = T,
        workers=4)
  
  # Set up grid for plotting
  grid_x <- seq(-2,2,by = 0.01)
  grid_h1 <- 0.3*exp(-4*(grid_x + 1)^2) + 0.7*exp(-16*(grid_x - 1)^2)
  grid_x2 <- rt(n=401, df=4)
  grid_x1 <- 0.5*grid_x + rnorm(401)
  grid_y <- 2*grid_x1 + grid_x2 + grid_h1
  
  x_grid_l <- array_reshape(as.matrix(cbind(grid_x1, grid_x2)), c(401, 2))
  x_grid_nl <- array_reshape(as.matrix(grid_x), c(401, 1))
  
  # Predict and isolate nonlinear component
  pred_grid <- model_hybrid %>%
    predict(list(x_grid_nl, x_grid_l))
  
  weighted_mat <- as.matrix(cbind(grid_x1, grid_x2)) %*% as.matrix(c(get_weights(model_hybrid)[[5]][6:7]))
  nonlinear_comp <- pred_grid - weighted_mat
  
  plr_matrix <- as.matrix(input_df_tmp[1:train, ])
  plr_matrix[,4] <- plr_matrix[,4] + 2.000000000002
  
  # Fit model
  plm_model <- plrm.est(plr_matrix)
  nonparametric_estimate <- plm_model$m.t
  nonparametric_fit_plm <- cbind(nonparametric_estimate, plr_matrix[,4])
  nonparametric_fit_plm <- nonparametric_fit_plm[order(nonparametric_fit_plm[,2]), ]
  
  # Store results
  save(grid_x, file = './code/r_code/results_simulation/results_semiparametric_model/visualisation_approximation/grid_x')
  save(grid_h1, file = './code/r_code/results_simulation/results_semiparametric_model/visualisation_approximation/grid_h1')
  save(nonlinear_comp, file = './code/r_code/results_simulation/results_semiparametric_model/visualisation_approximation/nonlinear_comp_ann')
  save(nonparametric_fit_plm, file = './code/r_code/results_simulation/results_semiparametric_model/visualisation_approximation/nonparametric_fit_plm')
}

#### Visualisation ####

# Cacluclate IMSE
# Prepare data
size=1000/0.8
set.seed(27)
x_true <- runif(size, -2,2)[1:1000]
h_true <- 0.3*exp(-4*(x_true + 1)^2) + 0.7*exp(-16*(x_true - 1)^2)

averaged_prediction_ann <- estimation_nonparametric_ann %>%
  summarise_all(funs(mean))

averaged_prediction_plm <- estimation_nonparametric_plm %>%
  summarise_all(funs(mean))

df_bias_tmp_ann <- sweep(as.matrix(averaged_prediction_ann[,1:1000]),2,h_true)
df_bias_ann <- data.frame(df_bias_tmp_ann^2)

df_bias_tmp_plm <- sweep(as.matrix(averaged_prediction_plm[,1:1000]),2,h_true)
df_bias_plm <- data.frame(df_bias_tmp_plm^2)

int_sq_bias_ann <- integrate.xy(x = x_true, df_bias_ann[1,])
int_sq_bias_plm <- integrate.xy(x = x_true, df_bias_plm[1,])

bias_vec <- c(NA, NA, NA, round(int_sq_bias_ann, 4), round(int_sq_bias_plm,4))

## Integrated Variance
average_pred_ann <- unname(unlist(averaged_prediction_ann[,1:1000]))
average_pred_plm <- unname(unlist(averaged_prediction_plm[,1:1000]))

df_variance_ann <- sweep(as.matrix(estimation_nonparametric_ann[,1:1000]),2,average_pred_ann)
df_variance_plm <- sweep(as.matrix(estimation_nonparametric_plm[,1:1000]),2,average_pred_plm)

df_variance_ann <- df_variance_ann^2
df_variance_plm <- df_variance_plm^2

variance_ann <- colMeans(df_variance_ann)
variance_plm <- colMeans(df_variance_plm)

int_var_ann <- integrate.xy(x = x_true, variance_ann)
int_var_plm <- integrate.xy(x = x_true, variance_plm)

variance_vec <- c(NA, NA, NA, round(int_var_ann, 4), round(int_var_plm,4))

# MSPE

mean_linear <- round(mean(sqrt(mse_dataframe$linear)),4)
mean_semi <- round(mean(sqrt(mse_dataframe$hybrid)),4)
mean_ann <- round(mean(sqrt(mse_dataframe$ann_pure)),4)
mean_ideal <- round(mean(sqrt(ideal_mse$MSE)),4)

result_table <- matrix(c(mean_ideal, mean_linear, mean_ann, mean_semi, NA),
                  ncol=1, nrow = 5)
result_table <- cbind(result_table, bias_vec, variance_vec)

colnames(result_table) <- c('RMSPE', 'int bias','int_variance')
rownames(result_table) <- c('Truth', 'Linear', 'ANN', 'SANN', 'Kernel')
stargazer(result_table, digits = 4)

## Visualisation approximation 

png('./tex_files/thesis/Figures/r_figures/fit_no_kernel_vs_ann', 
    width = 631, height = 369)
plot(grid_x, grid_h1, type='l', lwd=2, ylim = c(-0.1, 0.8),
     xlab=expression('x'[3]),
     ylab="")
mtext(expression(paste('g(','x'[3],')')),side=2,las=1,line=1,
      adj = 1.59)
lines(grid_x, nonlinear_comp, lwd=2, lty='dashed', col='red')
lines(nonparametric_fit_plm[,2]-2.000000000002, nonparametric_fit_plm[,1], col='green', lty='dotted', lwd=2)
legend('topright',
       legend = c('Truth', 'ANN', 'Kernel'),
       col=c('black', 'red', 'green'),
       lty=1:3,
       lwd=2)
dev.off()
