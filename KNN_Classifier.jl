using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, CSV, MLJ, DataFrames, Random, NearestNeighborModels
include("./Data_Processing.jl")

#Load the data
drop = generate(option = "drop", std = "false", valid = "false", test = "true");
med = generate(option = "med", std = "false", valid = "false", test = "true");

Random.seed!(2711)
#Function to tune the model with K values between 1 and 50
function TunedModel_KNN(data)
    model = KNNClassifier()
    self_tuning_model = TunedModel(model = model,
                                resampling = CV(nfolds = 10),
                                tuning = Grid(),
                                range = range(model, :K, values = 1:50),
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data, Not(:precipitation_nextday)),
                            data.precipitation_nextday) |> fit!
    self_tuning_mach
end

#Test the tune model on the drop data set and med data set:

rep1 = report(TunedModel_KNN(drop.train));
scatter(reshape(rep1.plotting.parameter_values, :), rep1.plotting.measurements, xlabel = "K", ylabel = "AUC")
# K = 27 with drop

rep2 = report(TunedModel_KNN(med.train));
scatter(reshape(rep2.plotting.parameter_values, :), rep2.plotting.measurements, xlabel = "K", ylabel = "AUC")
# K = 25 with med

# Test the best model over the test set with the best hyper-param top pick the best one
best_mach1 = machine(KNNClassifier(K = rep1.best_model.K), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!;
best_mach2 = machine(KNNClassifier(K = rep2.best_model.K), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday)|> fit!;
AUC1 = losses(best_mach1, select(drop.test, Not(:precipitation_nextday)), drop.test.precipitation_nextday)
AUC2 = losses(best_mach2, select(med.test, Not(:precipitation_nextday)), med.test.precipitation_nextday)
# AUC_drop = 0.904 , AUC_med = 0.898

#Write in the submission file with a machine trained on all data
train, test = generate(option = "drop", std = "false", valid = "false", test = "false");
if AUC1.auc > AUC2.auc
    best_mach = machine(KNNClassifier(K = rep1.best_model.K), select(train[:,:], Not(:precipitation_nextday)), train.precipitation_nextday)|> fit!;
else
    best_mach = machine(KNNClassifier(K = rep2.best_model.K), select(train[:,:], Not(:precipitation_nextday)), train.precipitation_nextday)|> fit!;
end
sample = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
pred = pdf.(predict(best_mach, test), true);
sample.precipitation_nextday = pred;
CSV.write(joinpath(@__DIR__, "results", "KNN_submission.csv"), sample);