using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, Plots, CSV, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux
include("./Data_Processing.jl")

train_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);
drop, drop_std, med, med_std = generate(train_data, "false");
coerce!(train_data, :precipitation_nextday => Multiclass);
coerce!(test_data, :precipitation_nextday => Multiclass);
drop

#Simple MLP with similar layers.
#mach1 = machine(NeuralNetworkClassifier(builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu), Dense(100, n_out))),
#     batch_size = 32, epochs = 20), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;

mach2 = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 2000, dropout = 0.1, σ = sigmoid),
      batch_size = 1132, epochs = 200), select(drop.train[:,:], Not(:precipitation_nextday)), drop.train.precipitation_nextday) |> fit!;
mean(predict_mode(mach2, select(drop.test[:,:], Not(:precipitation_nextday))) .!= drop.test.precipitation_nextday)
auc_m = MLJ.auc(predict(mach2, select(drop.test[:,:], Not(:precipitation_nextday))), drop.test.precipitation_nextday)


#fit!(m_mlp, verbosity = 2)
"""
mean(predict_mode(m_mlp, test_input) .!= test_output)



model = @pipeline(Standardizer(),
                       NeuralNetworkClassifier(
                             builder = MLJFlux.Short(n_hidden = 150,
                                                     σ = sigmoid),
                             optimiser = ADAM(),
                             batch_size = 32),
                             target = Standardizer())

tuned_model = TunedModel(model = model,
							  resampling = CV(nfolds = 5),
	                          range = [range(model,
						                :(neural_network_classifier.builder.dropout),
									    values = [0., .1, .2]),
								       range(model,
									     :(neural_network_classifier.epochs),
									     values = [500, 1000, 2000])],
	                          measure = auc)

mach = fit!(machine(tuned_model,
	                     train_input,
		                 train_output))
"""

drop_all, drop_std_all = generate_all(train_data, "drop");
best_mach = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 2000, dropout = 0.1, σ = sigmoid),
      batch_size = 3000, epochs = 200), select(drop_all[:,:], Not(:precipitation_nextday)), drop_all.precipitation_nextday) |> fit!;
pred = pdf.(predict(best_mach, test_data), true)
example_submission.precipitation_nextday = pred
CSV.write(joinpath(@__DIR__,"MLP_submission.csv"), example_submission)