using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Plots, CSV, StatsPlots, MLJ, DataFrames, MLJLinearModels, Random, Flux, MLJFlux

weather = CSV.read(joinpath(@__DIR__, "data","trainingdata.csv"), DataFrame)
test = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

imputed_data = MLJ.transform(fit!(machine(FillImputer(), select(weather[:,:], Not([:precipitation_nextday])))), select(weather[:,:], Not([:precipitation_nextday])));
data = insertcols!(imputed_data,:precipitation_nextday=>weather.precipitation_nextday[:]);
coerce!(data, :precipitation_nextday => Multiclass);
coerce!(data, Count => MLJ.Continuous);

train_input = select(data[1:3000,:], Not(:precipitation_nextday))
train_output = data.precipitation_nextday[1:3000]
test_input = select(data[3001:3176,:], Not(:precipitation_nextday))
test_output = data.precipitation_nextday[3001:3176]

m_mlp = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, sigmoid),
                                                          Dense(100, n_out))),
                         batch_size = 15,
                         epochs = 20),
                         train_input,
                         train_output)

fit!(m_mlp, verbosity = 2)

mean(predict_mode(m_mlp, test_input) .!= test_output)

auc_mlp = MLJ.auc(predict(m_mlp, test_input), test_output)


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
