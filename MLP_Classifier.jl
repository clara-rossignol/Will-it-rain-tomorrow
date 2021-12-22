using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux, MLJXGBoostInterface
include("./Data_Processing.jl")

#Load the data
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
drop = generate(option = "drop", std = "false", valid = "false", test = "true");
med = generate(option = "med", std = "false", valid = "false", test = "true");

#First scheme of possible neural network on med data set and drop data set, standardized or not.
NN_1 = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.1, σ = sigmoid),
                                                batch_size = 1132, epochs = 100), 
                                                select(med.train[:,:], Not(:precipitation_nextday)),
                                                med.train.precipitation_nextday) |> fit!;

AUC_drop = MLJ.auc(predict(NN_1, select(med.test[:,:], Not(:precipitation_nextday))), med.test.precipitation_nextday)
# D&S = 0.902, D&L = 0.909, M&S = 0.907, M&L = 0.903

#Second attempt
NN_2 = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.1, σ = relu),
                                                batch_size = 1132, epochs = 100), 
                                                select(drop.train[:,:], Not(:precipitation_nextday)),
                                                drop.train.precipitation_nextday) |> fit!;

AUC_drop = MLJ.auc(predict(NN_2, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)
# D&S = 0.916, D&L = 0.919, M&S = 0.891, M&L = 0.911

#Third attempt
model = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.1, σ = relu), batch_size = 50, epochs = 20)
tuned_model = TunedModel(model = model,
                        resampling = CV(nfolds = 20),
                        range = [range(model,
                        :(builder.dropout),
                        values = [0., .1, .2]),
                              range(model,
                              :epochs,
                              values = [500, 1000, 2000])],
                        measure = auc)
NN_3 = machine(tuned_model, select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;
rep = report(NN_3)
rep.best_model
rep.best_history_entry.measurement

AUC_drop = MLJ.auc(predict(rep.best_model, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)
# D&S = 0.916, D&L = 0.919, M&S = 0.891, M&L = 0.911

drop = generate(option = "drop", std = "true", valid = "false", test = "true");

#Fourth attempt
NN_4 = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 1000, dropout = 0.1, σ = relu),
                                                batch_size = 1500, epochs = 300), 
                                                select(drop.train[:,:], Not(:precipitation_nextday)),
                                                drop.train.precipitation_nextday) |> fit!;

AUC_drop = MLJ.auc(predict(NN_4, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)
# D&S = 0.916