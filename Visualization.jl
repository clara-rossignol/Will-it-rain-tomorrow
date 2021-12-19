using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, MLJ, DataFrames

weather = CSV.read(joinpath(@__DIR__, "data","trainingdata.csv"), DataFrame);
test = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);