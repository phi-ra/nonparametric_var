##------------------------------------------##
##  Author: Philipp Ratz
##  Title:  Motivating Example and 
##          overfitting behaviour of ANNs
##  Year:   2019
##  Tested: apple-darwin
##------------------------------------------##
rm(list=ls())
gc()

source('./code/r_code/helper_functions/lagged_data_motivating.R')
library(keras)
library(forecast)

# Model Parameters
N <- 400
lags <- 4
series_length <- N + lags

set.seed(42)
y_0 <- runif(1)

# Initialize observation vector
y_vec <- numeric(series_length)
y_vec[1] <- y_0
y_linear <- y_vec
y_nonlinear <- y_vec

for (pos in 2:length(y_vec)) {
  y_vec[pos] <- 0.3*y_vec[pos-1] + (22/pi)*sin(2*pi*y_vec[pos-1])
}

y_vec <- scale(y_vec)

for (pos in 2:length(y_vec)) {
  y_linear[pos] <- 0.3*y_vec[pos-1]
  y_nonlinear[pos] <- (22/pi)*sin(2*pi*y_vec[pos-1])
}

## Fit ARIMA
arma_model <- auto.arima(y_vec)
arma_in_sample <- forecast(arma_model)

## Fit ANN
data_train <- get_lagged_data(lags = 4, input_data = y_vec)

network <- keras_model_sequential() %>%
  layer_dense(units = 50, activation = "sigmoid", input_shape = c(lags)) %>% 
  layer_dense(units = 1, activation = 'linear')

network %>% compile(
  optimizer = optimizer_adam(),
  loss = "mse"
)

network %>%
  fit(data_train[[1]], data_train[[2]],
      epochs = 250, steps_per_epoch = 75,
      workers=n_cores, shuffle = F, verbose = T)

y_hat <- network %>%
  predict(data_train[[1]])

png('./tex_files/thesis/Figures/r_figures/motivating_example.png', 
    width = 936, height = 542)
plot(y_vec[200:250], type='l', lwd=2, axes=FALSE, xlab="Timesteps", ylab="")
lines(y_hat[196:246], lwd=2, lty='dotted', col='chartreuse3')
lines(arma_in_sample$fitted[200:250], col='darkorange2', lty='dashed', lwd=2)
lines(y_linear[200:250], lwd=2, lty='dashed')
axis(1, labels = FALSE)
axis(2, labels=TRUE)
box()
legend('topright',
       legend = c('Truth', 'Linear Component', 'ARMA', 'ANN'),
       col=c('black', 'black','darkorange2', 'chartreuse3'),
       lty=c(1,2,2,3),
       lwd=2)
dev.off()

