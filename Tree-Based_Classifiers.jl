using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux, MLJXGBoostInterface, MLJDecisionTreeInterface
include("./Data_Processing.jl")

train_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
coerce!(train_data, :precipitation_nextday => Multiclass);
coerce!(test_data, :precipitation_nextday => Multiclass);

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
    model_forest = RandomForestClassifier()
    self_tuning_model = TunedModel(model = model_forest,
                                resampling = CV(nfolds = 6),
                                tuning = Grid(),
                                range = range(model_forest, :n_trees, lower = 100, upper = 1000, scale = :log),
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data.train, Not(:precipitation_nextday)),
                            data.train.precipitation_nextday) |> fit!
    self_tuning_mach
end

#rep1 = report(TunedModel_forest(drop))
rep2 = report(TunedModel_forest(med))
#rep1.best_history_entry.measurement
rep2.best_history_entry.measurement

med_all = generate_all(train_data, "med");
best_mach = machine(RandomForestClassifier(n_trees = rep2.best_model.n_trees), select(med_all[:,:], Not(:precipitation_nextday)), med_all.precipitation_nextday)|> fit!

#To write in the submission file
pred = pdf.(predict(best_mach, test_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"RandomForest_submission.csv"), example_submission)



#-------------------------------------------------------------------------------#
#XGBoostClassifier

function TunedModel_XGB(data)
    model_xgb = XGBoostClassifier()
    self_tuning_model = TunedModel(model = model_xgb,
                                resampling = CV(nfolds = 6),
                                tuning = Grid(goal = 20),
                                range = [range(model_xgb, :eta,
                                        lower = 1e-4, upper = .1, scale = :log),
                                 range(model_xgb, :num_round, lower = 50, upper = 500),
                                 range(model_xgb, :max_depth, lower = 2, upper = 6)],
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data.train, Not(:precipitation_nextday)),
                            data.train.precipitation_nextday) |> fit!
    self_tuning_mach
end

#better with med than drop
rep1 = report(TunedModel_XGB(drop))
rep2 = report(TunedModel_XGB(med))
rep1.best_history_entry.measurement
rep2.best_history_entry.measurement


med_all = generate_all(train_data, "med");
best_mach = machine(XGBoostClassifier(eta = rep2.best_model.eta, num_round = rep2.best_model.num_round, max_depth = rep2.best_model.max_depth), 
                    select(med_all[:,:], Not(:precipitation_nextday)), med_all.precipitation_nextday)|> fit!

#To write in the submission file
pred = pdf.(predict(best_mach, test_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"xgb_submission.csv.csv"), example_submission)
