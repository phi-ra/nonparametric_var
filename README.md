# Thesis on semi-nonparametric forecasting

This repository contains the code for the masters thesis 'Semi-Nonparametric Forecasting via ANN-sieves'.

The pdf-version of the thesis is available as well as the code.

The codes should work as is, but there are doubtless still some bugs present. A large part of the simulations is parallelized to make them more efficient. Still, running all files will take about a week (depending on your hardware even longer). To make things slightly more complicated, three different computers were used for the simulations. The operating system, where the file was run is indicated at the top of each file and add some more information below.

To avoid issues when saving results, `cd` into the ./Thesis directory and run
```
chmod +x folder_creation.sh 
./folder_creation.sh
```
this will create all folders and subfolders where the results can be stored. 

In time, I hope to provide the results and the different algorithms in form of a package, but this might take a while. If you have and comments or questions, feel free to contribute. 

| Name in files        | Information       | Keras version  | Tensorflow version|
| - | - | - | - |
| Apple-Darwin    | R version 3.5.1 (2018-07-02) |  keras_2.2.0 | 1.0.0 |
|| x86_64-apple-darwin15.6.0 (64-bit) |
| Windows >=8   | R version 3.5.3 (2019-01-11) | keras_2.2.4    | 1.13.1 |
|| x86_64-w64-mingw32/x64 (64-bit) |
| Linux Mint | R version 3.4.4 (2018-03-15) | keras_2.2.4.1 | 1.13.1 | 
|| x86_64-pc-linux-gnu (64-bit) |
