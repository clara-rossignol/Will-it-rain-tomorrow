using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, MLJ, DataFrames, MLJLinearModels, Random, MLJLIBSVMInterface
import GLMNet: glmnet
include("./Data_Processing.jl")

train_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
drop, drop_std, med, med_std = generate(train_data, "false");
coerce!(train_data, :precipitation_nextday => Multiclass);
coerce!(test_data, :precipitation_nextday => Multiclass);

# Test if the predictor is better fitted on drop or median data
mach_drop = machine(LogisticClassifier(), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;
mach_med = machine(LogisticClassifier(), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday) |> fit!;

losses(mach_drop, select(drop.test[:,:], Not(:precipitation_nextday)), drop.test.precipitation_nextday) # 0.786
losses(mach_med, select(med.test[:,:], Not(:precipitation_nextday)), med.test.precipitation_nextday) # 0.882  

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

# Drop using lasso seems to have the best AUC.

#-------------------------------------------------------------------------------#
#Seconde part
drop, drop_std, med, med_std = generate(weather, "false");

function TunedModel_Reg(data)
	self_tuning_model = TunedModel(model = LogisticClassifier(),
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal = 100),
	                         range = range(model = LogisticClassifier(),
							 			:lambda, scale = :log,
									    lower = 1e-3, upper = 1e2),
	                         measure = auc)
    self_tuning_mach = machine(self_tuning_model, select(data.train[:,:], Not(:precipitation_nextday)), data.train.precipitation_nextday) |> fit!;
	self_tuning_mach
end

rep = report(TunedModel_Reg(drop))
rep.best_model
scatter(reshape(rep.plotting.parameter_values, :), rep.plotting.measurements, xlabel = "Lamba", ylabel = "AUC")

"""
best_mach = machine(KNNClassifier(K = 27), select(drop.train, Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!
losses(best_mach, select(drop.test[:,:], Not(:precipitation_nextday)), drop.test.precipitation_nextday)

To write in the submission file
pred = pdf.(predict(m_KNN, test), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"submission.csv"), example_submission)
"""