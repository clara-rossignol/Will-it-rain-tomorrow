using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, OpenML, CSV, StatsPlots, MLJ, DataFrames, MLJLinearModels, Random, MLJLIBSVMInterface, NearestNeighborModels

weather = CSV.read(joinpath(@__DIR__, "data","trainingdata.csv"), DataFrame)
test = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

size(weather)
describe(weather)

weather_no_missing = dropmissing(weather)
size(weather_no_missing)

"""
Here it is not a good idea to use the function dropmissing (removes rows with missing values) 
as it's removing almost half of the data !
We use FillIputer, which imputes missing data with a fixed value computed on the non-missing 
values. --> Try to change which value it computes
"""

imputed_data = MLJ.transform(fit!(machine(FillImputer(), select(weather[:,:], Not([:precipitation_nextday])))), select(weather[:,:], Not([:precipitation_nextday])));

data = insertcols!(imputed_data,:precipitation_nextday=>weather.precipitation_nextday[:]);

coerce!(data, :precipitation_nextday => Multiclass);

train_input = select(data[1:3000,:], Not(:precipitation_nextday))
train_output = data.precipitation_nextday[1:3000]
test_input = select(data[3001:3176,:], Not(:precipitation_nextday))
test_output = data.precipitation_nextday[3001:3176]

m_logistic = machine(LogisticClassifier(penalty = :l2, lambda = 2e-2), train_input, train_output) |> fit!;

# should put functions in a function file 
function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response)
	)
end;

losses(m_logistic, data[3001:3176,:], test_output)

confusion_matrix(predict_mode(m_logistic, test_input), test_output)


