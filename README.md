# Will it rain tomorrow?

## Introduction
This repository contains a machine learning approach to solve the "Will it rain tomorrow ?" problem. The task is to predict whether there will be some precipitation (rain, snow etc.) during the next day in Pully given some measurements at different weather stations in Switzerland.

The repository is constituted of:
- A data folder with three datasets from the problem; A training set, a test set to make prediction, and a example of a submission file.
- A results folder where the submissions will appear when running the code. The submissions are under the .csv format.
- A visualization Pluto notebook where we explore the data.
- Multiple .jl files for each type of model built in order to solve this problem. In each .jl file you can observe how we processed in order to develop our best model for a particular type of model.
- A license.
- A .gitignore in order to avoid IDE originating files and results files.
- A .toml file with all the necessary packages needed to run the code.
- A report where we review in more detail each part of the project and talk a bit about the problem and datasets. The report is under the .pdf format.


------------------
Data_Processing.jl
------------------

When using multiple ML methods, the data must undergo multiple changes. It is rapidly confusing to execute every time each transformation. In order to solve this problem, we implemented a generate(; option, std, valid, test) function. This function has many modes:
- option = "drop" or "med": the first option will drop the rows containing missing data and the second will fill out the missing data with the median of the predictor.
- std = "false" or "true": the returned data is standardized or not.
- valid = "true" or "false": the returned data contains a training set, a validation set and a test set. We add automatically a test set because the other case is equivalent to putting test = "true" (see next arg description).
- test = "true" or "false": the returned data contains a training and a test set.

Special scenario: in order to train every model on the full training set and make predictions on the test set, we need to transform both sets with the same machine, for example when using standardization. When using this combination of arg generate(; option, std, valid = "false", test = "false") and any value for option and std, the function returns the training and test set transformed with the same machine in order to train a given model on the full training set and to predict with this model on the test set.


-------------------------
Running a particular model
-------------------------
For each type of model we tried, we implemented a .jl file named with the type of model it corresponds to. In order to construct a prediction .csv file with a specific model, you only need to run the associated .jl file. In each file, the code is similar:
- First, we include all the needed packages to run the code.
- We load each dataset.
- We test some small model in order to rapidly exclude some sizes of hyper-parameters or some transformation of the data.
- Then, we tune a model over a particular range for some hyper-parameters.
- We test a few of the best hyper-parameters over our own test set.
- We pick the best model we found and use it to make predictions on the given test set.
- We generate the .csv file in the results folder.


---------------------------------
Running the visualization notebook
---------------------------------
In order to run the Pluto notebook "Visualization_Notebook.jl" one must download [julia](https://julialang.org/downloads) (at least version 1.6.2) and open the julia terminal.

```julia
julia> using Pkg
       Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       using MLCourse
       using Pluto
       Pluto.run(notebook = "path_to_notebook")
```

## Authors
- [Oscar Henry](https://github.com/Oscar-Henry)
- [Clara Rossignol](https://github.com/clara-rossignol)


![](https://www.epfl.ch/wp/5.5/wp-content/themes/wp-theme-2018/assets/svg/epfl-logo.svg)
