using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux, MLJXGBoostInterface, Plots
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
                        range = range(model, :epochs,
                        values = [500, 1000, 2000]),
                        measure = auc)
NN_3 = machine(tuned_model, select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday);
fit!(NN_3, verbosity = 2)
rep = report(NN_3)
rep.best_model
rep.best_history_entry.measurement

AUC_drop = MLJ.auc(predict(rep.best_model, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)
# D&S = 0.916, D&L = 0.919, M&S = 0.891, M&L = 0.911

drop = generate(option = "drop", std = "true", valid = "false", test = "true");

#Fourth attempt
NN_4 = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.1, σ = relu),
                                                batch_size = 1000, epochs = 1000), 
                                                select(drop.train[:,:], Not(:precipitation_nextday)),
                                                drop.train.precipitation_nextday);
fit!(NN_4, verbosity = 2)
AUC_drop = MLJ.auc(predict(NN_4, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)
# D&S = 0.910


#Fifth attempt
function TunedModel_NN(data)
    model = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 200, dropout = 0.1, σ = relu), batch_size = 50)
    self_tuning_model = TunedModel(model = model,
                                resampling = CV(nfolds = 20),
                                tuning = Grid(),
                                range = [range(model, :epochs,
                                        lower = 10, upper = 2000, scale = :log),
                                 range(model, :batch_size, lower = 10, upper = 1000, scale = :log),
                                 range(model, :(builder.dropout), values = [0., 0.1])],
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data, Not(:precipitation_nextday)),
                            data.precipitation_nextday) |> fit!
    self_tuning_mach
end

drop = generate(option = "drop", std = "true", valid = "false", test = "true");
rep1 = report(TunedModel_NN(drop.train))
rep1.best_history_entry.measurement

#Attempt 6
model = NeuralNetworkClassifier( builder = MLJFlux.Short(n_hidden = 1000, dropout = 0.2,
			σ = NNlib.σ),
	finaliser = NNlib.softmax,
	optimiser = ADAM(0.001, (0.9, 0.999), IdDict{Any,Any}()),
	loss = Flux.crossentropy,
	epochs = 300,  #150
	batch_size = 2700,
	lambda = 0.0,
	alpha = 0.0,
	optimiser_changes_trigger_retraining = false)

result=[]
for _ in 1:10
    NN_6 = machine(model, select(drop.train, Not(:precipitation_nextday)), drop.train.precipitation_nextday)
    fit!(NN_6)
    AUC_drop = MLJ.auc(predict(NN_6, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)
    push!(result, AUC_drop)
    model.epochs = model.epochs + 50
end
scatter(1:10, result)


AUC_drop = MLJ.auc(predict(NN_6, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)

#Write in the submission file with a machine trained on all data, Drop data set and K=27 seems to have the highest AUC on the test data
tot = generate(option = "drop", std = "true", valid = "false", test = "false");
NN_6 = machine(model, select(tot, Not(:precipitation_nextday)), tot.precipitation_nextday)
fit!(NN_6)
Random.seed!(2809)
std_mach = machine(Standardizer(), select(test_data, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
fit!(std_mach)
new_data = MLJ.transform(std_mach, select(test_data, Not([:ZER_sunshine_1, :ALT_sunshine_4])))

pred = pdf.(predict(NN_6, new_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"MLP_submission.csv"), example_submission)