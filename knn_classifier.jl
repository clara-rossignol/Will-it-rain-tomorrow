using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, CSV, MLJ, DataFrames, Random, NearestNeighborModels
include("./Data_Processing.jl")

train_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
drop, drop_std, med, med_std = generate(train_data, "false");
coerce!(train_data, :precipitation_nextday => Multiclass);
coerce!(test_data, :precipitation_nextday => Multiclass);

function TunedModel_KNN(data)
    model_KNN = KNNClassifier()
    self_tuning_model = TunedModel(model = KNNClassifier(),
                                resampling = CV(nfolds = 10),
                                tuning = Grid(),
                                range = range(model = KNNClassifier(), :K, values = 5:50),
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data.train, Not(:precipitation_nextday)),
                            data.train.precipitation_nextday) |> fit!
    self_tuning_mach
end

rep1 = report(TunedModel_KNN(drop))
rep2 = report(TunedModel_KNN(med))
rep1.best_history_entry.measurement
rep2.best_history_entry.measurement
scatter(reshape(rep2.plotting.parameter_values, :), rep2.plotting.measurements, xlabel = "K", ylabel = "AUC")

med_all, med_std_all = generate_all(train_data, "med");
best_mach = machine(KNNClassifier(K = rep2.best_model.K), select(med_all[:,:], Not(:precipitation_nextday)), med_all.precipitation_nextday)|> fit!

#To write in the submission file
pred = pdf.(predict(best_mach, test_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"KNN_submission.csv"), example_submission)