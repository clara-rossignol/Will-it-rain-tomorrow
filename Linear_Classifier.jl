using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, MLJ, DataFrames, MLJLinearModels, Random, MLJLIBSVMInterface
import GLMNet: glmnet
include("./Data_Processing.jl")

#-------------------------------------------------------------------------------#
#First part

drop, drop_std, med, med_std = generate(weather, valid = "true");
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

mach1 = machine(LogisticClassifier(penalty = :none), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;
# 0.757
mach2 = machine(LogisticClassifier(penalty = :l2, lambda = 2e-2), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;
# 0.832
mach3 = machine(LogisticClassifier(penalty = :none), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday) |> fit!;
# 0.759
mach4 = machine(LogisticClassifier(penalty = :l2, lambda = 2e-2), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday) |> fit!;
# 0.811
#mach5 = machine(LogisticClassifier(penalty = :l1, lambda = 2e-2), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;
#           

losses(mach1, select(drop.test[:,:], Not(:precipitation_nextday)), drop.test.precipitation_nextday)
losses(mach2, select(drop.test[:,:], Not(:precipitation_nextday)), drop.test.precipitation_nextday)
losses(mach3, select(med.test[:,:], Not(:precipitation_nextday)), med.test.precipitation_nextday)
losses(mach4, select(med.test[:,:], Not(:precipitation_nextday)), med.test.precipitation_nextday)

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