using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, StatsPlots, MLJ, DataFrames, MLJLinearModels, Random, MLJLIBSVMInterface, NearestNeighborModels
import GLMNet: glmnet

example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

train = CSV.read(joinpath(@__DIR__, "data","train.csv"), DataFrame)
valid = CSV.read(joinpath(@__DIR__, "data","valid.csv"), DataFrame)
test = CSV.read(joinpath(@__DIR__, "data","test.csv"), DataFrame)

coerce!(train, :precipitation_nextday => Multiclass);
coerce!(valid, :precipitation_nextday => Multiclass);
coerce!(test, :precipitation_nextday => Multiclass);

m_logistic = machine(LogisticClassifier(penalty = :l2, lambda = 2e-2), select(train[:,:], Not([:precipitation_nextday])), train.precipitation_nextday) |> fit!;

# should put functions in a function file 
function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response)
	)
end;

function tune_model(model, data)
	tuned_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 20),
	                         tuning = Grid(goal = 200),
	                         range = range(model, :lambda,
									       scale = :log,
									       lower = 1e-3, upper = 1e2),
	                         measure = auc)
                             machine(LogisticClassifier(penalty = :l2, lambda = 2e-2), select(train[:,:], Not([:precipitation_nextday])), train.precipitation_nextday) |> fit!;
end

res1_lasso = fitted_params(tune_model(LogisticClassifier(), train))

"""losses(m_logistic, select(test[:,:], Not([:precipitation_nextday])), test.precipitation_nextday)"""

"""confusion_matrix(predict_mode(m_logistic, select(test[:,:], Not([:precipitation_nextday]))), test.precipitation_nextday)"""


