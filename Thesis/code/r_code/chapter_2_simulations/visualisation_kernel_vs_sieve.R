##---------------------------------------------##
##  Author: Philipp Ratz
##  Title:  Comparison asymptotic convergence 
##          sieve - kernel type regression
##  Year:   2019
##  Tested: apple-darwin
##---------------------------------------------##
rm(list=ls())
gc()

# Load packages
library(reshape2)
library(plotly)

# Helper functions
dimensionality_kernel <- function(data_size, dimensionality, order=2) {
  converge <- data_size^((-2*order)/(2*order+dimensionality))
  return(converge)
}
dimensionality_sieve <- function(data_size, dimensionality) {
  converge <- (data_size/log(data_size))^(-(1+2/(1+dimensionality))/(4*(1+1/(1+dimensionality))))
  return(converge)
}

# Set up grid data
data_size <- seq(1, 1e4, 1e2)
dims <- seq(1,100, 1)

kernel_matrix <- matrix(NA, nrow = 100, ncol = 100)
sieve_matrix <- matrix(NA, nrow = 100, ncol = 100)

for (data_sizer in seq(1e4, 1e6, by = 1e4)) {
  kernel_matrix[data_sizer/1e4, ] <- dimensionality_kernel(data_size = data_sizer, dimensionality = dims)
  sieve_matrix[data_sizer/1e4, ] <- dimensionality_sieve(data_size = data_sizer, dimensionality = dims)
}

# Visualize in plotly, force to adapt a single color per surface with a little hack
color <- rep(0, length(kernel_matrix))
dim(color) <- dim(kernel_matrix)
color2 <- rep(1, length(kernel_matrix))
dim(color2) <- dim(kernel_matrix)

plot_ly(colors = c('gray18', 'gray70')) %>%
  add_surface(x=dims,
              y=data_size,
              z=sieve_matrix,
              opacity = 1,
              surfacecolor=color,
              cauto=F,
              cmax=1,
              cmin=0,
              name='Sieve',
              showscale=F,
              showlegend=T) %>%
  add_surface(x=dims,
              y=data_size,
              z=kernel_matrix,
              opacity = 1,
              surfacecolor=color2,
              cauto=F,
              cmax=1,
              cmin=0,
              name='Kernel',
              showscale=F) %>%
  layout(scene=list(xaxis=list(title='Dimensions'),
                    yaxis=list(title='N'),
                    zaxis=list(title='O-Term')
  )
  )
## Export manually due to the bug with plotly
dev.off()
