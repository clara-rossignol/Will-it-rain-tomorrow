using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Plots
include("./Data_Processing.jl")

train_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

drop = generate(train_data, option = "drop", valid = "false", test = "true");
med = generate(train_data, option = "med", valid = "false", test = "true");


# Test if the predictor is better fitted on drop or median data
mach_drop = machine(LogisticClassifier(), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;
mach_med = machine(LogisticClassifier(), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday) |> fit!;

losses(mach_drop, select(drop.test[:,:], Not(:precipitation_nextday)), drop.test.precipitation_nextday) # 0.82
losses(mach_med, select(med.test[:,:], Not(:precipitation_nextday)), med.test.precipitation_nextday) # 0.88


#-------------------------------------------------------------------------------#

function TunedModel_Reg(data)
	model = LogisticClassifier()
	self_tuning_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal = 100),
	                         range = range(model,
							 			:lambda,
									    lower = 1e-4, upper = 1e2, scale = :log),
	                         measure = auc)
    self_tuning_mach = machine(self_tuning_model, 
							select(data[:,:], Not(:precipitation_nextday)), 
							data.precipitation_nextday) |> fit!;
	self_tuning_mach
end

rep1 = report(TunedModel_Reg(drop.train))
rep1.best_history_entry.measurement
rep1.best_model.lambda
scatter(reshape(rep1.plotting.parameter_values, :), rep1.plotting.measurements, xlabel = "Lamba", ylabel = "AUC")

rep2 = report(TunedModel_Reg(med.train))
rep2.best_history_entry.measurement
rep2.best_model.lambda
scatter(reshape(rep2.plotting.parameter_values, :), rep2.plotting.measurements, xlabel = "Lamba", ylabel = "AUC")

best_mach1 = machine(LogisticClassifier(lambda = rep1.best_model.lambda), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!
best_mach2 = machine(LogisticClassifier(lambda = rep2.best_model.lambda), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday)|> fit!

losses(best_mach1, select(drop.test, Not(:precipitation_nextday)), drop.test.precipitation_nextday)
losses(best_mach2, select(med.test, Not(:precipitation_nextday)), med.test.precipitation_nextday)


#Write in the submission file with a machine trained on all data
tot = generate(train_data, option = "med", valid = "false", test = "false");
best_mach = machine(LogisticClassifier(lambda = rep2.best_model.lambda), select(tot[:,:], Not(:precipitation_nextday)), tot.precipitation_nextday) |> fit!

pred = pdf.(predict(best_mach, test_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"Linear_submission.csv"), example_submission)




#jsp ce que c'est ça
"""
model = @pipeline(Standardizer(),
                    LogisticClassifier(),
                    target = Standardizer())
tuned_model = TunedModel(model = model,
				resampling = CV(nfolds = 5),
				tuning = Grid(goal = 100),
				range = range(model = model,
							:lambda, scale = :log,
							 lower = 1e-3, upper = 1e2),
	            measure = auc)
mach2 = fit!(machine(tuned_model,
	                     select(drop.train, Not(:precipitation_nextday)),
		                 ,drop.train.precipitation_nextday)
mach2
"""
