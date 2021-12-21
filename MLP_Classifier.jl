using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux, MLJXGBoostInterface
include("./Data_Processing.jl")

train_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
coerce!(train_data, :precipitation_nextday => Multiclass);
coerce!(test_data, :precipitation_nextday => Multiclass);

example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

drop = generate(train_data, option = "drop", valid = "false", test = "true");
med = generate(train_data, option = "med", valid = "false", test = "true");



m_nn_drop = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.1, σ = sigmoid),
      batch_size = 1132, epochs = 100), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;
mean(predict_mode(m_nn_drop, select(drop.test[:,:], Not(:precipitation_nextday))) .!= drop.test.precipitation_nextday)
auc_m_nn_drop = MLJ.auc(predict(m_nn_drop, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)

m_nn_med = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.1, σ = sigmoid),
      batch_size = 2000, epochs = 100), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday) |> fit!;
mean(predict_mode(m_nn_med, select(med.test[:,:], Not(:precipitation_nextday))) .!= med.test.precipitation_nextday)
auc_m_nn_med = MLJ.auc(predict(m_nn_med, select(med.test[:,:], Not(:precipitation_nextday))), med.test.precipitation_nextday)




m_mlp = machine(NeuralNetworkClassifier(
                    builder = MLJFlux.@builder(Chain(Dense(n_in, 500, sigmoid), Dense(500, n_out))),
                    batch_size = 1000,
                    epochs = 100),
                    select(drop.train[:,:], Not(:precipitation_nextday)), 
                    drop.train.precipitation_nextday) |> fit!
mean(predict_mode(m_mlp, select(drop.test[:,:], Not(:precipitation_nextday))) .!= drop.test.precipitation_nextday)
auc_m = MLJ.auc(predict(m_mlp, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)

m_mlp = machine(NeuralNetworkClassifier(
                    builder = MLJFlux.@builder(Chain(Dense(n_in, 500, sigmoid), Dense(500, n_out))),
                    optimiser = ADAMW(),
                    batch_size = 2117,
                    epochs = 100),
                    select(med.train[:,:], Not(:precipitation_nextday)), 
                    med.train.precipitation_nextday) |> fit!
mean(predict_mode(m_mlp, select(med.test[:,:], Not(:precipitation_nextday))) .!= med.test.precipitation_nextday)
auc_m = MLJ.auc(predict(m_mlp, select(med.test[:,:], Not(:precipitation_nextday))), med.test.precipitation_nextday)




#doesn't work and I can't figure out why
"""
m_h = machine(NeuralNetworkClassifier(
                    builder = MLJFlux.MLP(hidden=(10,)), #3 hidden layers with 10 neurons each
                    batch_size = 1200,
                    epochs = 100),
                    select(drop.train[:,:], Not(:precipitation_nextday)), 
                    drop.train.precipitation_nextday) |> fit!
mean(predict_mode(m_h, select(drop.test[:,:], Not(:precipitation_nextday))) .!= drop.test.precipitation_nextday)
auc_mh = MLJ.auc(predict(m_h, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)
"""


