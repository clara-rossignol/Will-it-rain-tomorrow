using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux, MLJXGBoostInterface, MLJDecisionTreeInterface
include("./Data_Processing.jl")

#Load the data
drop_std = generate(option = "drop", std = "true", valid = "false", test = "true");
med_std = generate(option = "med", std = "true", valid = "false", test = "true");
drop = generate(option = "drop", std = "false", valid = "false", test = "true");
med = generate(option = "med", std = "false", valid = "false", test = "true");

#Random Forest Classifier
m_forest_drop = machine(RandomForestClassifier(n_trees = 500), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit! 
m_forest_med = machine(RandomForestClassifier(n_trees = 1200), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday) |> fit!
m_forest_drop_std = machine(RandomForestClassifier(n_trees = 500), select(drop_std.train[:,:], Not(:precipitation_nextday)), drop_std.train.precipitation_nextday) |> fit!
m_forest_med_std = machine(RandomForestClassifier(n_trees = 1200), select(med_std.train[:,:], Not(:precipitation_nextday)), med_std.train.precipitation_nextday) |> fit!

auc_forest_drop = MLJ.auc(predict(m_forest_drop, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday) # 0.909
auc_forest_med = MLJ.auc(predict(m_forest_med, select(med.test[:,:], Not(:precipitation_nextday))), med.test.precipitation_nextday) # 0.927
auc_forest_drop_std = MLJ.auc(predict(m_forest_drop_std, select(drop_std.test[:,:], Not(:precipitation_nextday))), drop_std.test.precipitation_nextday) # 0.908
auc_forest_med_std = MLJ.auc(predict(m_forest_med_std, select(med_std.test[:,:], Not(:precipitation_nextday))), med_std.test.precipitation_nextday) # 0.926


#The med data seems to have the best AUC (not a big difference between standardized or not)


function TunedModel_forest(data)
    Random.seed!(2711)
    model = RandomForestClassifier()
    self_tuning_model = TunedModel(model = model,
                                resampling = CV(nfolds = 5),
                                tuning = Grid(),
                                range = range(model, :n_trees, values=100:50:1200),
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data, Not(:precipitation_nextday)),
                            data.precipitation_nextday) |> fit!
    self_tuning_mach
end


#Test the tuned model on the drop data set and med data set with both standardized and not-standardized data:
rep1 = report(TunedModel_forest(drop_std.train));
rep1.best_history_entry.measurement
rep1.best_model.n_trees
scatter(reshape(rep1.plotting.parameter_values, :), rep1.plotting.measurements, xlabel = "N", ylabel = "AUC")

rep2 = report(TunedModel_forest(med_std.train));
rep2.best_history_entry.measurement
rep2.best_model.n_trees
scatter(reshape(rep2.plotting.parameter_values, :), rep2.plotting.measurements, xlabel = "N", ylabel = "AUC")

rep3 = report(TunedModel_forest(drop.train));
rep3.best_history_entry.measurement
rep3.best_model.n_trees
scatter(reshape(rep3.plotting.parameter_values, :), rep3.plotting.measurements, xlabel = "N", ylabel = "AUC")

rep4 = report(TunedModel_forest(med.train));
rep4.best_history_entry.measurement 
rep4.best_model.n_trees
scatter(reshape(rep4.plotting.parameter_values, :), rep4.plotting.measurements, xlabel = "N", ylabel = "AUC")



#Test the best model over the test set with the best hyper-parameters to pick the best one:
best_mach1 = machine(RandomForestClassifier(n_trees = rep1.best_model.n_trees), select(drop_std.train[:,:], Not(:precipitation_nextday)), drop_std.train.precipitation_nextday)|> fit!
best_mach2 = machine(RandomForestClassifier(n_trees = rep2.best_model.n_trees), select(med_std.train[:,:], Not(:precipitation_nextday)), med_std.train.precipitation_nextday)|> fit!
best_mach3 = machine(RandomForestClassifier(n_trees = rep3.best_model.n_trees), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!
best_mach4 = machine(RandomForestClassifier(n_trees = rep4.best_model.n_trees), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday)|> fit!
losses(best_mach1, select(drop_std.test, Not(:precipitation_nextday)), drop_std.test.precipitation_nextday)
losses(best_mach2, select(med_std.test, Not(:precipitation_nextday)), med_std.test.precipitation_nextday)
losses(best_mach3, select(drop.test, Not(:precipitation_nextday)), drop.test.precipitation_nextday)
losses(best_mach4, select(med.test, Not(:precipitation_nextday)), med.test.precipitation_nextday)


#Write in the submission file with the best machine trained on all data
train, test = generate(option = "med", std = "true", valid = "false", test = "false");
best_mach = machine(RandomForestClassifier(n_trees = rep2.best_model.n_trees), select(train[:,:], Not(:precipitation_nextday)), train.precipitation_nextday) |> fit!

sample = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
pred = pdf.(predict(best_mach, test), true);
sample.precipitation_nextday = pred;
CSV.write(joinpath(@__DIR__, "results", "RandomForest_submission.csv"), sample);