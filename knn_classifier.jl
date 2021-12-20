using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, CSV, MLJ, DataFrames, MLJLinearModels, Random, NearestNeighborModels

weather = CSV.read(joinpath(@__DIR__, "data","trainingdata.csv"), DataFrame)
test = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

imputed_data = MLJ.transform(fit!(machine(FillImputer(), select(weather[:,:], Not([:precipitation_nextday])))), select(weather[:,:], Not([:precipitation_nextday])));
data = insertcols!(imputed_data,:precipitation_nextday=>weather.precipitation_nextday[:]);
coerce!(data, :precipitation_nextday => Multiclass);

idxs = randperm(size(data, 1))
rows = size(data)[1]
indices = Vector{Int}([1,floor(rows/10),floor(2*rows/10),floor(rows)])

train = data[idxs[indices[1]:indices[2]], :]
valid = data[idxs[indices[2]:indices[3]], :]
test = data[idxs[indices[3]:indices[4]], :]

train_input = select(train[:,:], Not(:precipitation_nextday))
train_output = train.precipitation_nextday
test_input = select(test[:,:], Not(:precipitation_nextday))
test_output = test.precipitation_nextday


function TunedModel_KNN(data)
    model_KNN = KNNClassifier()
    self_tuning_model = TunedModel(model = model_KNN,
                                resampling = CV(nfolds = 20),
                                tuning = Grid(),
                                range = range(model_KNN, :K, values = 1:100),
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data, Not(:precipitation_nextday)),
                            data.precipitation_nextday) |> fit!
    
    #scatter(reshape(rep.plotting.parameter_values, :),
	    #rep.plotting.measurements, xlabel = "K", ylabel = "AUC")
    report(self_tuning_mach).best_model.K
end

best_KNN_machine = machine(KNNClassifier(K = TunedModel_KNN(train)), train_input, train_output) |>fit!
auc_KNN = MLJ.auc(predict(best_KNN_machine, test_input), test_output)

""" To write in the submission file
pred = pdf.(predict(m_KNN, test), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"submission.csv"), example_submission)
"""