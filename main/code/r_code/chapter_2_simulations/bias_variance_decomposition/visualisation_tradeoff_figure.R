##-----------------------------------------##
##  Author: Philipp Ratz
##  Title:  Visualisation Bias Variance
##          Decomposition
##  Year:   2019
##  Tested: apple-darwin
##-----------------------------------------##
rm(list=ls())
gc()

# Load packages, data and regenerate truth
library(dplyr)
library(reshape2)
source('./code/r_code/helper_functions/data_generation.R')

file_singlelayer_tradeoff <- './code/r_code/results_simulation/results_bias_variance/results_singlelayer/tradeoff_data_single.csv'

if(file.exists(file_singlelayer_tradeoff)) {
 df <- read.csv('./code/r_code/results_simulation/results_bias_variance/results_singlelayer/tradeoff_data_single.csv',
                sep=';')
 colnames(df)[1] <- 'Hidden_units'
} else {
  print('File not yet generated please run simulation_bias_variance.R first')
}

# Regenerate truth
set.seed(42)
truth  <- generate_data_iid(size = 400, mean=0, sd=256,
                        x_from=-10, x_to = 10, break_at = 1)[[3]]
x <- generate_data_iid(size = 400, mean=0, sd=256,
                   x_from=-10, x_to = 10, break_at = 1)[[1]]

## Bias
averaged_prediction <- df %>%
  group_by(Hidden_units) %>%
  summarise_all(funs(mean))

df_bias_tmp <- sweep(as.matrix(averaged_prediction[,2:402]),2,truth[1:401])
df_bias <- data.frame(df_bias_tmp^2)
avg_bias <- rowMeans(df_bias)

## Variance

hidden_units <- unique(df$Hidden_units)
prepared_variance <- matrix(nrow = 1, ncol = 402)

for (units in hidden_units) {
  unit_df <- df %>%
    filter(Hidden_units == units) %>%
    dplyr::select(-one_of('Hidden_units'))
  
  predictor_sweep <- sweep(as.matrix(unit_df), 2,
                           as.matrix(averaged_prediction[which(averaged_prediction$Hidden_units==units),
                                                         2:402]))
  var_all <- predictor_sweep^2
  var_all <- cbind(rep(units, dim(var_all)[1]), var_all)
  
  prepared_variance <- rbind(prepared_variance, var_all)
}

var_df <- data.frame(prepared_variance[2:dim(prepared_variance)[1],])

var_final <- var_df %>%
  group_by(V1) %>%
  summarise_all(funs(mean))

avg_var <- rowMeans(var_final)

## MSE
mse_all <- sweep(as.matrix(df[,2:402]),2,truth[1:401])
mse_all <- mse_all^2

hu <- df$Hidden_units
bind_df <- cbind(hu, data.frame(mse_all))
df_mse <- bind_df %>% 
  group_by(hu) %>%
  summarise_all(funs(mean))
avg_mse <- rowMeans(df_mse)

#### Visualisation ####
x <- c(1,seq(10, 730, by = 30))

lo_bias <- loess(avg_bias[1:26]~x, degree = 1)
lo_var <- loess(avg_var[1:26]~x,  degree = 1)
lo_mse <- loess(avg_mse[1:26]~x,  degree = 1)

png('./tex_files/thesis/Figures/r_figures/bias_variance_tradeoff_ann.png', 
    width = 701, height = 469)
plot(x, predict(lo_bias), type='l', xlab='Hidden units', ylab="", lwd=2, axes=FALSE)
lines(x, predict(lo_var), lwd=2, lty='dashed')
lines(x, predict(lo_mse), lwd=2, lty='dotted')
axis(1, labels = TRUE)
axis(2, labels=FALSE)
box()
legend('topright',
       legend = c(expression(Bias^2), expression(Var[e]), 'MSE'),
       col=c('black', 'black', 'black'),
       lty=1:3,
       lwd=2)
dev.off()
