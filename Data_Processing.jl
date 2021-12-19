using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, MLJ, DataFrames

weather = CSV.read(joinpath(@__DIR__, "data","trainingdata.csv"), DataFrame);

describe(weather)

data_drop = dropmissing(weather);
size(weather)

data_med = MLJ.transform(fit!(machine(FillImputer(), select(weather[:,:], Not([:precipitation_nextday])))),
                             select(weather[:,:], Not([:precipitation_nextday])));
insertcols!(data_med,:precipitation_nextday=>weather.precipitation_nextday[:]);

size(data_drop)
size(data_med)

coerce!(data_drop, :precipitation_nextday => Multiclass);
coerce!(data_med, :precipitation_nextday => Multiclass);

data_drop_std = data_drop;
data_med_std = data_med;

idxs1 = randperm(size(data_drop, 1));
idxs2 = randperm(size(data_med, 1));
dim_drop = size(data_drop)[1];
dim_med = size(data_med)[1];

idx_drop = Vector{Int}([1, floor(dim_drop/10), floor(2*dim_drop/10), dim_drop]);
#1 / 169 / 339 / 1699
idx_med = Vector{Int}([1, floor(dim_med/10), floor(2*dim_med/10), dim_med]);
#1 / 317 / 653 / 3176

final_drop = (train = data_drop[idxs1[idx_drop[1]:idx_drop[2]], :], valid = data_drop[idxs1[idx_drop[2]:idx_drop[3]], :], test = data_drop[idxs1[idx_drop[3]:idx_drop[4]], :]);
final_drop_std = (train = data_drop_std[idxs1[idx_drop[1]:idx_drop[2]], :], valid = data_drop_std[idxs1[idx_drop[2]:idx_drop[3]], :], test = data_drop_std[idxs1[idx_drop[3]:idx_drop[4]], :]);
final_med = (train = data_med[idxs2[idx_med[1]:idx_med[2]], :], valid = data_med[idxs2[idx_med[2]:idx_med[3]], :], test = data_med[idxs2[idx_med[3]:idx_med[4]], :]);
final_med_std = (train = data_med_std[idxs2[idx_med[1]:idx_med[2]], :], valid = data_med_std[idxs2[idx_med[2]:idx_med[3]], :], test = data_med_std[idxs2[idx_med[3]:idx_med[4]], :]);

CSV.write(joinpath(@__DIR__, "data/Data_drop.csv"), final_drop);
CSV.write(joinpath(@__DIR__, "data/Data_drop_std.csv"), final_drop_std);
CSV.write(joinpath(@__DIR__, "data/Data_med.csv"), final_med);
CSV.write(joinpath(@__DIR__, "data/Data_med_std.csv"), final_med_std);

"""