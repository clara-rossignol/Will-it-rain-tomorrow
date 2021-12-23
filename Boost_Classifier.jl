using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, CSV, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux, MLJXGBoostInterface, MLJDecisionTreeInterface
include("./Data_Processing.jl")

#Load the data
drop_std = generate(option = "drop", std = "true", valid = "false", test = "true");
med_std = generate(option = "med", std = "true", valid = "false", test = "true");
drop = generate(option = "drop", std = "false", valid = "false", test = "true");
med = generate(option = "med", std = "false", valid = "false", test = "true");

#Function to tune the model with hyper-parameters eta, num_round, and max_depth
function TunedModel_XGB(data)
    Random.seed!(2711)
    model = XGBoostClassifier()
    self_tuning_model = TunedModel(model = model,
                                resampling = CV(nfolds = 10),
                                tuning = Grid(goal = 100),
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


#Test the tuned model on the drop data set and med data set with both standardized and not-standardized data:
rep1 = report(TunedModel_XGB(drop_std.train));
rep1.best_history_entry.measurement

rep2 = report(TunedModel_XGB(med_std.train));
rep2.best_history_entry.measurement

rep3 = report(TunedModel_XGB(drop.train));
rep3.best_history_entry.measurement #AUC for dropped, non standardized data 0.9176; best hyper-parameters being eta = 0.1, num_round = 400, max_depth = 4

rep4 = report(TunedModel_XGB(med.train));
rep4.best_history_entry.measurement #AUC for dropped, non standardized data 0.9297; best hyper-parameters being eta = 0.1, num_round = 310, max_depth = 6


#Test the best model over the test set with the best hyper-parameters to pick the best one:
best_mach1 = machine(XGBoostClassifier(eta = rep1.best_model.eta, num_round = rep1.best_model.num_round, max_depth = rep1.best_model.max_depth), select(drop_std.train[:,:], Not(:precipitation_nextday)), drop_std.train.precipitation_nextday)|> fit!
best_mach2 = machine(XGBoostClassifier(eta = rep2.best_model.eta, num_round = rep2.best_model.num_round, max_depth = rep2.best_model.max_depth), select(med_std.train[:,:], Not(:precipitation_nextday)), med_std.train.precipitation_nextday)|> fit!
best_mach3 = machine(XGBoostClassifier(eta = rep3.best_model.eta, num_round = rep3.best_model.num_round, max_depth = rep3.best_model.max_depth), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!
best_mach4 = machine(XGBoostClassifier(eta = rep4.best_model.eta, num_round = rep4.best_model.num_round, max_depth = rep4.best_model.max_depth), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday)|> fit!
losses(best_mach1, select(drop_std.test, Not(:precipitation_nextday)), drop_std.test.precipitation_nextday)
losses(best_mach2, select(med_std.test, Not(:precipitation_nextday)), med_std.test.precipitation_nextday)
losses(best_mach3, select(drop.test, Not(:precipitation_nextday)), drop.test.precipitation_nextday)
losses(best_mach4, select(med.test, Not(:precipitation_nextday)), med.test.precipitation_nextday)

#med with standardized data seems to perform the best


#Write in the submission file with the best machine trained on all data
train, test = generate(option = "med", std = "true", valid = "false", test = "false");
best_mach = machine(XGBoostClassifier(eta = rep2.best_model.eta, num_round = rep2.best_model.num_round, max_depth = rep2.best_model.max_depth), select(train[:,:], Not(:precipitation_nextday)), train.precipitation_nextday)|> fit!
sample = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);m
pred = pdf.(predict(best_mach, test), true);
sample.precipitation_nextday = pred;
CSV.write(joinpath(@__DIR__, "results", "XGBoost_submission.csv"), sample);