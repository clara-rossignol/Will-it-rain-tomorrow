using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux, MLJXGBoostInterface, MLJDecisionTreeInterface
include("./Data_Processing.jl")

train_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

drop = generate(train_data, option = "drop", valid = "false", test = "true");
med = generate(train_data, option = "med", valid = "false", test = "true");


#-------------------------------------------------------------------------------#
#Random Forest Classifier

m_forest_drop = machine(RandomForestClassifier(n_trees = 500), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!
auc_forest_drop = MLJ.auc(predict(m_forest_drop, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)

m_forest_med = machine(RandomForestClassifier(n_trees = 500), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday) |> fit!
auc_forest_med = MLJ.auc(predict(m_forest_med, select(med.test[:,:], Not(:precipitation_nextday))), med.test.precipitation_nextday)

#med seems to have the best AUC

function TunedModel_forest(data)
    model = RandomForestClassifier()
    self_tuning_model = TunedModel(model = model,
                                resampling = CV(nfolds = 6),
                                tuning = Grid(),
                                range = range(model, :n_trees, lower = 100, upper = 1000, scale = :log),
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data, Not(:precipitation_nextday)),
                            data.precipitation_nextday) |> fit!
    self_tuning_mach
end

drop = generate(train_data, option = "drop", valid = "false", test = "true");
rep1 = report(TunedModel_forest(drop.train))
rep1.best_history_entry.measurement
rep1.best_model.n_trees

med = generate(train_data, option = "med", valid = "false", test = "true");
rep2 = report(TunedModel_forest(med.train))
rep2.best_history_entry.measurement
rep2.best_model.n_trees

best_mach1 = machine(RandomForestClassifier(n_trees = rep1.best_model.n_trees), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!
best_mach2 = machine(RandomForestClassifier(n_trees = rep2.best_model.n_trees), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday)|> fit!

losses(best_mach1, select(drop.test, Not(:precipitation_nextday)), drop.test.precipitation_nextday)
losses(best_mach2, select(med.test, Not(:precipitation_nextday)), med.test.precipitation_nextday)

#Write in the submission file with a machine trained on all data
tot = generate(train_data, option = "med", valid = "false", test = "false");
best_mach = machine(RandomForestClassifier(n_trees = rep2.best_model.n_trees), select(tot[:,:], Not(:precipitation_nextday)), tot.precipitation_nextday) |> fit!

pred = pdf.(predict(best_mach, test_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"RandomForest_submission.csv"), example_submission)




#-------------------------------------------------------------------------------#
#XGBoostClassifier

function TunedModel_XGB(data)
    model = XGBoostClassifier()
    self_tuning_model = TunedModel(model = model,
                                resampling = CV(nfolds = 6),
                                tuning = Grid(),
                                range = [range(model, :eta,
                                        lower = 1e-4, upper = .1, scale = :log),
                                 range(model, :num_round, lower = 50, upper = 500),
                                 range(model, :max_depth, lower = 2, upper = 6)],
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data, Not(:precipitation_nextday)),
                            data.precipitation_nextday) |> fit!
    self_tuning_mach
end


model = XGBoostClassifier()

drop = generate(train_data, option = "drop", valid = "false", test = "true");
rep1 = report(TunedModel_XGB(drop.train))
rep1.best_history_entry.measurement

med = generate(train_data, option = "med", valid = "false", test = "true");
rep2 = report(TunedModel_XGB(med.train))
rep2.best_history_entry.measurement


best_mach1 = machine(XGBoostClassifier(eta = rep1.best_model.eta, num_round = rep1.best_model.num_round, max_depth = rep1.best_model.max_depth), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!
best_mach2 = machine(XGBoostClassifier(eta = rep2.best_model.eta, num_round = rep2.best_model.num_round, max_depth = rep2.best_model.max_depth), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday)|> fit!

losses(best_mach1, select(drop.test, Not(:precipitation_nextday)), drop.test.precipitation_nextday)
losses(best_mach2, select(med.test, Not(:precipitation_nextday)), med.test.precipitation_nextday)

#Write in the submission file with a machine trained on all data
tot = generate(train_data, option = "med", valid = "false", test = "false");
best_mach = machine(XGBoostClassifier(eta = rep2.best_model.eta, num_round = rep2.best_model.num_round, max_depth = rep2.best_model.max_depth), select(tot[:,:], Not(:precipitation_nextday)), tot.precipitation_nextday)|> fit!

pred = pdf.(predict(best_mach, test_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"XGB_submission.csv"), example_submission)
