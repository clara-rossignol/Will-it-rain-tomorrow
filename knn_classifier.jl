using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, OpenML, CSV, StatsPlots, MLJ, DataFrames, MLJLinearModels, Random, MLJLIBSVMInterface, NearestNeighborModels

weather = CSV.read(joinpath(@__DIR__, "data","trainingdata.csv"), DataFrame)
test = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

imputed_data = MLJ.transform(fit!(machine(FillImputer(), select(weather[:,:], Not([:precipitation_nextday])))), select(weather[:,:], Not([:precipitation_nextday])));
data = insertcols!(imputed_data,:precipitation_nextday=>weather.precipitation_nextday[:]);
coerce!(data, :precipitation_nextday => Multiclass);

train_input = select(data[1:3000,:], Not(:precipitation_nextday))
train_output = data.precipitation_nextday[1:3000]
test_input = select(data[3001:3176,:], Not(:precipitation_nextday))
test_output = data.precipitation_nextday[3001:3176]

m_KNN = machine(KNNClassifier(K = 36), train_input, train_output) |>fit!

err = mean(predict_mode(m_KNN, test_input) .!= test_output)

auc_KNN = MLJ.auc(predict(m_KNN, test_input), test_output)

pred = pdf.(predict(m_KNN, test), true)

example_submission.precipitation_nextday = pred

CSV.write(joinpath(@__DIR__,"submission.csv"), example_submission)


model_KNN = KNNClassifier()
self_tuning_model = TunedModel(model = model_KNN,
                                resampling = CV(nfolds = 5),
                                tuning = Grid(),
                                range = range(model_KNN, :K, values = 1:50),
                                measure = auc)
self_tuning_mach = machine(self_tuning_model,
                            select(data, Not(:precipitation_nextday)),
                            data.precipitation_nextday) |> fit!

rep = report(self_tuning_mach)

scatter(reshape(rep.plotting.parameter_values, :),
	    rep.plotting.measurements, xlabel = "K", ylabel = "AUC")
