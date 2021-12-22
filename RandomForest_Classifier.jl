using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux, MLJXGBoostInterface, MLJDecisionTreeInterface
include("./Data_Processing.jl")

#Load the data
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
drop = generate(option = "drop", std = "true", valid = "false", test = "true");
med = generate(option = "med", std = "true", valid = "false", test = "true");

#Random Forest Classifier
m_forest_drop = machine(RandomForestClassifier(n_trees = 500), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!
m_forest_med = machine(RandomForestClassifier(n_trees = 1000), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday) |> fit!

auc_forest_med = MLJ.auc(predict(m_forest_med, select(med.test[:,:], Not(:precipitation_nextday))), med.test.precipitation_nextday) # 0.919
auc_forest_drop = MLJ.auc(predict(m_forest_drop, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday) # 0.917

#The Med data set seems to have the best AUC
function TunedModel_forest(data)
    model = RandomForestClassifier()
    self_tuning_model = TunedModel(model = model,
                                resampling = CV(nfolds = 6),
                                tuning = Grid(),
                                range = range(model, :n_trees, values=100:50:1200),
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data, Not(:precipitation_nextday)),
                            data.precipitation_nextday) |> fit!
    self_tuning_mach
end

#Different AUC in function of the hyperparameters.
drop = generate(option = "drop", valid = "false", test = "true");
rep1 = report(TunedModel_forest(drop.train))
rep1.best_history_entry.measurement
rep1.best_model.n_trees
scatter(reshape(rep1.plotting.parameter_values, :), rep1.plotting.measurements, xlabel = "N", ylabel = "AUC")

#Different AUC in function of the hyperparameters.
med = generate(option = "med", valid = "false", test = "true");
rep2 = report(TunedModel_forest(med.train))
rep2.best_history_entry.measurement
rep2.best_model.n_trees

best_mach1 = machine(RandomForestClassifier(n_trees = rep1.best_model.n_trees), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!
best_mach2 = machine(RandomForestClassifier(n_trees = rep2.best_model.n_trees), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday)|> fit!
#Which on perform the best on the test set.
losses(best_mach1, select(drop.test, Not(:precipitation_nextday)), drop.test.precipitation_nextday)
losses(best_mach2, select(med.test, Not(:precipitation_nextday)), med.test.precipitation_nextday)

#Write in the submission file with a machine trained on all data
tot = generate(train_data, option = "med", valid = "false", test = "false");
best_mach = machine(RandomForestClassifier(n_trees = rep2.best_model.n_trees), select(tot[:,:], Not(:precipitation_nextday)), tot.precipitation_nextday) |> fit!

pred = pdf.(predict(best_mach, test_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"RandomForest_submission.csv"), example_submission)
