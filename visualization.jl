using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, MLJ, DataFrames

weather = CSV.read(joinpath(@__DIR__, "data","trainingdata.csv"), DataFrame);
test = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);

function data_final(data,reorganization)
    if reorganization == "dropmissing"
        data = dropmissing(data)
    elseif reorganization == "median"
        data = MLJ.transform(fit!(machine(FillImputer(), select(data[:,:], Not([:precipitation_nextday])))), select(data[:,:], Not([:precipitation_nextday])));
        insertcols!(data,:precipitation_nextday=>weather.precipitation_nextday[:]);
    end
    
    coerce!(data, :precipitation_nextday => Multiclass);
    MLJ.transform(fit!(machine(Standardizer(count = true),data)), data)

    idxs = randperm(size(data, 1))
    rows = size(data)[1]
    indices = Vector{Int}([1,floor(rows/10),floor(2*rows/10),floor(rows)])

    train = data[idxs[indices[1]:indices[2]], :]
    valid = data[idxs[indices[2]:indices[3]], :]
    test = data[idxs[indices[3]:indices[4]], :]
    
    CSV.write(joinpath(@__DIR__, "data/train.csv"), train)
    CSV.write(joinpath(@__DIR__, "data/valid.csv"), valid)
    CSV.write(joinpath(@__DIR__, "data/test.csv"), test)
end

data_final(weather,"median")


