# Nonparametric Value-at-Risk (VaR) Forecasting
## Via sieve estimation aka neural networks

This reposoitory contains the code for the article "Nonparametric Value-at-Risk via Sieve Estimation"

> **_NOTE:_** Currently, some code is being rewritten to work with GPU to speed up the operations - there might be the occasional hickup

A .pdf file of the (work in progress) article will be added

The codes should work as is, but there are doubtless still some bugs present. A large part of the simulations is parallelized to make them more efficient. Still, running all files will take about a week (depending on your hardware even longer). To make things slightly more complicated, three different computers were used for the simulations. The operating system, where the file was run is indicated at the top of each file and add some more information below.

In time, I hope to provide the results and the different algorithms in form of a package, but this might take a while. If you have and comments or questions, feel free to contribute. 

| Name in files        | Information       | Keras version  | Tensorflow version|
| - | - | - | - |
| Apple-Darwin    | R version 3.5.1 (2018-07-02) |  keras_2.2.0 | 1.0.0 |
|| x86_64-apple-darwin15.6.0 (64-bit) |
| Windows >=8   | R version 3.5.3 (2019-01-11) | keras_2.2.4    | 1.13.1 |
|| x86_64-w64-mingw32/x64 (64-bit) |
| Linux Mint | R version 3.4.4 (2018-03-15) | keras_2.2.4.1 | 1.13.1 | 
|| x86_64-pc-linux-gnu (64-bit) |
