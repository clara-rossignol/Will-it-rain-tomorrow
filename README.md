# Will it rain tomorrow?

# Introduction:
This repository is a machine learning approach in order to solve the "will it rain tomorrow ?" problem.
The repository in constitued of:
- A data folder with the three data sets of the problem.
- A results folder where the sumbission will appears when running the code. The submission are under the .csv format.
- A visualization files where we explore the data using Pluto notebook.
- Multiple .jl files for each type of model build in order to solve this problem. In each .jl you can observe how we processed in order to develop our best model for a particular type of model.
- A license
- A .gitignore in order to avoid IDE originating files and results files.
- A .toml files with all the necessary packages needed to run the code
- A rapport where we review in more detail each part of the projects and talk a bit about the problem and data sets.

------------------
Data_Processing.jl
------------------

When using multiple ML methods, the data undergo multiple changes and it is rapidly confusing to execute every time every transormation. In order to solve this problem, we implemented a generate(; option, std, valid, test) function. This function have many modes which are;
- option = "drop" or "med", the first option will drop the missing data and the second will filled out the missing data with the median of the predictor.
- std = "false" or "true", the returned data is standardized or not
- valid = "true" or "false", the returned data contains a training set, a validation set and a test set. We add automatically a test set because the other case is equivalent a puting test = "true" (see next arg description).
- test = "true" or "false", the returned data contains a training and a test set.

Special scenario: In order to train every model one the full training set and make prediction on the test set, we need to transforme both set with the same machine, for example when using standardization. When using this combination of arg generate(; option, std, valid = "false", test = "false") and any value for option and std, the function return the training and test set transformed with the same machine in order to train a given model on the full training set and to predict with this model on the test set.

-------------------------
Runing a particular model
-------------------------
For each type of model we tried, we implemented a .jl files named with the tyoe of the model it correspond to. In order to construct a prediction .csv files with a specific model, you only need to run the associated .jl file. In each file, the code is similar:
- First, we include all the needed packages to run the code
- We load each data set.
- We test some small model in order to rapidly exclude some sizes of parameter or some transformation of the data.
- Then, we tuned a model over particular ranges for some hyperparameters.
- We test few of the best hyperparameters over our own test set.
- We pick the best model we found and use it to make prediction on the given test set.
- We generate the .csv file in the result folder.

Rapport: https://docs.google.com/document/d/1RPtgL8XIum415hB1V21glNvtt0mkM1hj2rOtG-8cjxs/edit?usp=sharing

The task is to predict whether there will be some precipitation (rain, snow etc.) on the next day in Pully given some measurements at different weather stations in Switzerland.


![](https://www.epfl.ch/wp/5.5/wp-content/themes/wp-theme-2018/assets/svg/epfl-logo.svg)
