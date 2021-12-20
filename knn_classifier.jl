using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, CSV, MLJ, DataFrames, MLJLinearModels, Random, NearestNeighborModels
include("./Data_Processing.jl")

drop, drop_std, med, med_std = generate(weather, "false");
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

function TunedModel_KNN(data)
    model_KNN = KNNClassifier()
    self_tuning_model = TunedModel(model = KNNClassifier(),
                                resampling = CV(nfolds = 10),
                                tuning = Grid(),
                                range = range(model_KNN, :K, values = 1:100),
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data.train, Not(:precipitation_nextday)),
                            data.train.precipitation_nextday) |> fit!
    self_tuning_mach
end

rep = report(TunedModel_KNN(drop))
scatter(reshape(rep.plotting.parameter_values, :), rep.plotting.measurements, xlabel = "Lamba", ylabel = "AUC")
rep.best_model
best_mach = machine(KNNClassifier(K = rep.best_model.K), select(drop.train, Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!
losses(best_mach, select(drop.test[:,:], Not(:precipitation_nextday)), drop.test.precipitation_nextday)

"""
best_KNN_machine1 = machine(KNNClassifier(K = TunedModel_KNN(drop.train)), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |>fit!
best_KNN_machine2 = machine(KNNClassifier(K = TunedModel_KNN(med.train)), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |>fit!
auc_KNN = MLJ.auc(predict(best_KNN_machine1, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)
auc_KNN = MLJ.auc(predict(best_KNN_machine2, select(med.test[:,:], Not(:precipitation_nextday))), med.test.precipitation_nextday)

To write in the submission file
pred = pdf.(predict(m_KNN, test), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"submission.csv"), example_submission)
"""