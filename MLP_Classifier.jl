using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux, MLJXGBoostInterface, Plots
include("./Data_Processing.jl")

#Load the data
drop = generate(option = "drop", std = "fale", valid = "false", test = "true");
med = generate(option = "med", std = "false", valid = "false", test = "true");
drop_std = generate(option = "drop", std = "true", valid = "false", test = "true");
med_std = generate(option = "med", std = "true", valid = "false", test = "true");

#First scheme of possible neural network on med data set and drop data set, standardized or not.
Random.seed!(2711);
NN_1 = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.1, σ = relu),
                                                batch_size = 1000, epochs = 100), 
                                                select(drop_std.train[:,:], Not(:precipitation_nextday)),
                                                drop_std.train.precipitation_nextday) |> fit!;

MLJ.auc(predict(NN_1, select(drop_std.test[:,:], Not(:precipitation_nextday))), drop_std.test.precipitation_nextday)
# AUC_DROP_STD = 0.912
# Relu seems to perform better than sigmoid function.
# Drop seems to perform better than med.
# No particular differences between standardized and normal data.

#Tuned model
#Use this section to try multiple model with changes in some hyper-parametrs in order to see if there is some increase in the AUC
Random.seed!(2711)
model = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 100, dropout = 0.1, σ = relu),
    epochs = 100,
    batch_size = 1000);

result=[];
for _ in 1:10 #Change the second number if you want to try some changes
    NN_2 = machine(model, select(drop_std.train, Not(:precipitation_nextday)), drop_std.train.precipitation_nextday)
    fit!(NN_2)
    AUC_drop = MLJ.auc(predict(NN_2, select(drop_std.test[:,:], Not(:precipitation_nextday))), drop_std.test.precipitation_nextday)
    push!(result, AUC_drop)
    model.builder.n_hidden = model.builder.n_hidden + 10
end
scatter(1:10, result, ylims = [0.8, 1.])

# Best model found in the previous part
Random.seed!(2711)
best_model = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 200, dropout = 0.1, σ = relu),
    epochs = 100,
    batch_size = 1000)
best_NN = machine(best_model, select(drop_std.train, Not(:precipitation_nextday)), drop_std.train.precipitation_nextday)
fit!(best_NN, verbosity = 2)
MLJ.auc(predict(best_NN, select(drop_std.test[:,:], Not(:precipitation_nextday))), drop_std.test.precipitation_nextday)

#Write in the submission file with a machine trained on all the data
train, test = generate(option = "drop", std = "true", valid = "false", test = "false");
best_mach = machine(best_model, select(train, Not(:precipitation_nextday)), train.precipitation_nextday);
fit!(best_mach)
sample = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
pred = pdf.(predict(best_mach, test), true);
sample.precipitation_nextday = pred;
CSV.write(joinpath(@__DIR__, "results", "MLP_submission.csv"), sample);
