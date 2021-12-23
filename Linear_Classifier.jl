using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Plots
include("./Data_Processing.jl")

#Load the data
drop = generate(option = "drop", std = "false", valid = "false", test = "true");
med = generate(option = "med", std = "false", valid = "false", test = "true");

#Test if the machine has a greater AUC on the drop data set or the med data set
mach_drop = machine(LogisticClassifier(), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;
mach_med = machine(LogisticClassifier(), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday) |> fit!;
losses(mach_drop, select(drop.test[:,:], Not(:precipitation_nextday)), drop.test.precipitation_nextday) # 0.79
losses(mach_med, select(med.test[:,:], Not(:precipitation_nextday)), med.test.precipitation_nextday) # 0.88
# AUC_drop = 0.786, AUC_med = 0.880

#Function to tune the model with L1 and L2 regularization with log scaled lambda values
function TunedModel_L2(data)
	Random.seed!(2711)
	model = LogisticClassifier(penalty = :l2)
	self_tuning_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal = 100),
	                         range = range(model, :lambda,
									    lower = 1e1, upper = 1e3, scale = :log),
	                         measure = auc)
    self_tuning_mach = machine(self_tuning_model, 
							select(data[:,:], Not(:precipitation_nextday)), 
							data.precipitation_nextday) |> fit!;
	self_tuning_mach
end

Random.seed!(2711)
function TunedModel_L1(data)
	model = LogisticClassifier(penalty = :l1)
	self_tuning_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal = 100),
	                         range = range(model, :lambda,
									    lower = 1e-1, upper = 1e2, scale = :log),
	                         measure = auc)
    self_tuning_mach = machine(self_tuning_model, 
							select(data[:,:], Not(:precipitation_nextday)), 
							data.precipitation_nextday) |> fit!;
	self_tuning_mach
end

#Load the standardized data
drop_std = generate(option = "drop", std = "true", valid = "false", test = "true");
med_std = generate(option = "med", std = "true", valid = "false", test = "true");

#Test de best model on the drop and med data set with L1 or L2 regularization
rep_drop_L2 = report(TunedModel_L2(drop_std.train));
scatter(reshape(rep_drop_L2.plotting.parameter_values, :), rep_drop_L2.plotting.measurements, xlabel = "Lamba", ylabel = "AUC")
rep_drop_L1 = report(TunedModel_L1(drop_std.train));
scatter(reshape(rep_drop_L1.plotting.parameter_values, :), rep_drop_L1.plotting.measurements, xlabel = "Lamba", ylabel = "AUC")
rep_med_L2 = report(TunedModel_L2(med_std.train));
scatter(reshape(rep_med_L2.plotting.parameter_values, :), rep_med_L2.plotting.measurements, xlabel = "Lamba", ylabel = "AUC")
rep_med_L1 = report(TunedModel_L1(med_std.train));
scatter(reshape(rep_med_L1.plotting.parameter_values, :), rep_med_L1.plotting.measurements, xlabel = "Lamba", ylabel = "AUC")

#Test the best models over the test set with the best hyper-parameters to pick the best one
best_mach_drop_L2 = machine(LogisticClassifier(penalty = :l2, lambda = rep_drop_L2.best_model.lambda), select(drop_std.train[:,:], Not(:precipitation_nextday)), drop_std.train.precipitation_nextday)|> fit!
best_mach_drop_L1 = machine(LogisticClassifier(penalty = :l1, lambda = rep_drop_L1.best_model.lambda), select(drop_std.train[:,:], Not(:precipitation_nextday)), drop_std.train.precipitation_nextday)|> fit!
best_mach_med_L2 = machine(LogisticClassifier(penalty = :l2, lambda = rep_med_L2.best_model.lambda), select(med_std.train[:,:], Not(:precipitation_nextday)), med_std.train.precipitation_nextday)|> fit!
best_mach_med_L1 = machine(LogisticClassifier(penalty = :l1, lambda = rep_med_L1.best_model.lambda), select(med_std.train[:,:], Not(:precipitation_nextday)), med_std.train.precipitation_nextday)|> fit!
losses(best_mach_drop_L2, select(drop_std.test, Not(:precipitation_nextday)), drop_std.test.precipitation_nextday)
losses(best_mach_drop_L1, select(drop_std.test, Not(:precipitation_nextday)), drop_std.test.precipitation_nextday)
losses(best_mach_med_L2, select(med_std.test, Not(:precipitation_nextday)), med_std.test.precipitation_nextday)
losses(best_mach_med_L1, select(med_std.test, Not(:precipitation_nextday)), med_std.test.precipitation_nextday)
# AUC_DROP_L2 = 0.917, AUC_DROP_L1 = 0.914, AUC_MED_L2 = 0.914, AUC_MED_L1 = 0.912

#Write in the submission file with the best machine trained on all the data
train, test = generate(option = "drop", std = "true", valid = "false", test = "false");
best_mach = machine(LogisticClassifier(penalty = :l2, lambda = rep_drop_L2.best_model.lambda), select(train[:,:], Not(:precipitation_nextday)), train.precipitation_nextday)|> fit!;
sample = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
pred = pdf.(predict(best_mach, test), true);
sample.precipitation_nextday = pred;
CSV.write(joinpath(@__DIR__, "results", "Linear_submission.csv"), sample);
