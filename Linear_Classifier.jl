using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Plots
include("./Data_Processing.jl")

#Load the data
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
drop = generate(option = "drop", std = "false", valid = "false", test = "true");
med = generate(option = "med", std = "false", valid = "false", test = "true");

# Test if the predictor is better fitted on drop or median data
mach_drop = machine(LogisticClassifier(), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;
mach_med = machine(LogisticClassifier(), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday) |> fit!;

losses(mach_drop, select(drop.test[:,:], Not(:precipitation_nextday)), drop.test.precipitation_nextday) # 0.79
losses(mach_med, select(med.test[:,:], Not(:precipitation_nextday)), med.test.precipitation_nextday) # 0.88

#The Med data set seems to perform better, now see if we can tune some rigde and lasso regularizator.

function TunedModel_L2(data)
	model = LogisticClassifier(penalty = :l2)
	self_tuning_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal = 100),
	                         range = range(model,
							 			:lambda,
									    lower = 1e2, upper = 1e3, scale = :log),
	                         measure = auc)
    self_tuning_mach = machine(self_tuning_model, 
							select(data[:,:], Not(:precipitation_nextday)), 
							data.precipitation_nextday) |> fit!;
	self_tuning_mach
end

function TunedModel_L1(data)
	model = LogisticClassifier(penalty = :l1)
	self_tuning_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal = 100),
	                         range = range(model,
							 			:lambda,
									    lower = 1e-1, upper = 1e2, scale = :log),
	                         measure = auc)
    self_tuning_mach = machine(self_tuning_model, 
							select(data[:,:], Not(:precipitation_nextday)), 
							data.precipitation_nextday) |> fit!;
	self_tuning_mach
end

#Load data
drop = generate(option = "drop", std = "true", valid = "false", test = "true");
med = generate(option = "med", std = "true", valid = "false", test = "true");

rep1 = report(TunedModel_L2(drop.train))
rep1.best_history_entry.measurement
rep1.best_model.lambda
#Show the different AUC in function of the hyperparameters. Lambda = 141
scatter(reshape(rep1.plotting.parameter_values, :), rep1.plotting.measurements, xlabel = "Lamba", ylabel = "AUC")

rep2 = report(TunedModel_L1(drop.train))
rep2.best_history_entry.measurement
rep2.best_model.lambda
#Show the different AUC in function of the hyperparameters. lambda = 9
scatter(reshape(rep2.plotting.parameter_values, :), rep2.plotting.measurements, xlabel = "Lamba", ylabel = "AUC")

# Drop is computed with K=27 and Med is computed with K=25
best_mach1 = machine(LogisticClassifier(penalty = :l2, lambda = rep1.best_model.lambda), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!
best_mach2 = machine(LogisticClassifier(penalty = :l1, lambda = rep2.best_model.lambda), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!

losses(best_mach1, select(drop.test, Not(:precipitation_nextday)), drop.test.precipitation_nextday)
losses(best_mach2, select(drop.test, Not(:precipitation_nextday)), drop.test.precipitation_nextday)


#Write in the submission file with a machine trained on all data
tot = generate(option = "med", std = "true", valid = "false", test = "false");
best_mach = machine(LogisticClassifier(lambda = rep2.best_model.lambda), select(tot[:,:], Not(:precipitation_nextday)), tot.precipitation_nextday) |> fit!

pred = pdf.(predict(best_mach, test_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"Linear_submission.csv"), example_submission)
