using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, CSV, MLJ, DataFrames, Random, NearestNeighborModels
include("./Data_Processing.jl")

#Load the data
train_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
coerce!(train_data, :precipitation_nextday => Multiclass);
coerce!(test_data, :precipitation_nextday => Multiclass);

#Function to tune the model
function TunedModel_KNN(data, model)
    model_KNN = KNNClassifier()
    self_tuning_model = TunedModel(model = model,
                                resampling = CV(nfolds = 10),
                                tuning = Grid(),
                                range = range(model, :K, values = 5:50),
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data, Not(:precipitation_nextday)),
                            data.precipitation_nextday) |> fit!
    self_tuning_mach
end

#Chose some transformation of the data.
model = KNNClassifier()

train_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
drop = generate(train_data, option = "drop", valid = "false", test = "true");
rep1 = report(TunedModel_KNN(drop.train, model))
rep1.best_history_entry.measurement
rep1.best_model.K
drop

train_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
med = generate(train_data, option = "med", valid = "false", test = "true");
rep2 = report(TunedModel_KNN(med.train, model))
rep2.best_history_entry.measurement
rep2.best_model.K
med

best_mach1 = machine(KNNClassifier(K = rep1.best_model.K), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday)|> fit!
best_mach2 = machine(KNNClassifier(K = rep2.best_model.K), select(med.train[:,:], Not(:precipitation_nextday)), med.train.precipitation_nextday)|> fit!

losses(best_mach1, select(drop.test, Not(:precipitation_nextday)), drop.test.precipitation_nextday)
losses(best_mach2, select(med.test, Not(:precipitation_nextday)), med.test.precipitation_nextday)


#Show the different AUC in function of the hyperparameters.
scatter(reshape(rep2.plotting.parameter_values, :), rep2.plotting.measurements, xlabel = "K", ylabel = "AUC")

#Write in the submission file with a machine trained on all data
med_all = generate(train_data, option = "med", valid = "false", test = "false");
med_all
best_mach = machine(KNNClassifier(K = rep2.best_model.K), select(med_all[:,:], Not(:precipitation_nextday)), med_all.precipitation_nextday)|> fit!

#To 
pred = pdf.(predict(best_mach, test_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"KNN_submission.csv"), example_submission)