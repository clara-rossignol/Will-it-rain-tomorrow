using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, MLJ, DataFrames, MLJLinearModels, MLJDecisionTreeInterface

weather = CSV.read(joinpath(@__DIR__, "data","trainingdata.csv"), DataFrame);
test = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

imputed_data = MLJ.transform(fit!(machine(FillImputer(), select(weather[:,:], Not([:precipitation_nextday])))), select(weather[:,:], Not([:precipitation_nextday])));

data = MLJ.transform(fit!(machine(Standardizer(count = true),imputed_data)), imputed_data)

insertcols!(data,:precipitation_nextday=>weather.precipitation_nextday[:]);

coerce!(data, :precipitation_nextday => Multiclass);

train_input = select(data[1:3000,:], Not(:precipitation_nextday))
train_output = data.precipitation_nextday[1:3000]
test_input = select(data[3001:3176,:], Not(:precipitation_nextday))
test_output = data.precipitation_nextday[3001:3176]

m_forest = machine(RandomForestClassifier(n_trees = 500), train_input, train_output) |> fit!

auc_forest = MLJ.auc(predict(m_forest, test_input), test_output)

mean(predict_mode(m_forest, test_input) .== test_output)

#evaluate!(m_forest, measure = rmse) does not work but can't figure out why

pred = pdf.(predict(m_forest, test), true)

example_submission.precipitation_nextday = pred

CSV.write(joinpath(@__DIR__,"submission.csv"), example_submission)



function TunedModel_forest(data)
    model_forest = RandomForestClassifier()
    self_tuning_model = TunedModel(model = model_forest,
                                resampling = CV(nfolds = 10),
                                tuning = Grid(),
                                range = range(model_forest, :n_trees, values = 100:800),
                                measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                            select(data, Not(:precipitation_nextday)),
                            data.precipitation_nextday) |> fit!
    rep = report(self_tuning_mach)
    rep.best_model
end


train = CSV.read(joinpath(@__DIR__, "data","train.csv"), DataFrame)
valid = CSV.read(joinpath(@__DIR__, "data","valid.csv"), DataFrame)
test = CSV.read(joinpath(@__DIR__, "data","test.csv"), DataFrame)

coerce!(train, :precipitation_nextday => Multiclass);
coerce!(valid, :precipitation_nextday => Multiclass);
coerce!(test, :precipitation_nextday => Multiclass);

TunedModel_forest(train)