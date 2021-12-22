### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ b0ded90e-6283-11ec-0039-2de048031f30
begin
	using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	using MLCourse, CSV, MLJ, DataFrames, StatsPlots, MLJMultivariateStatsInterface,
    Plots, LinearAlgebra, Statistics, Random, MLJLinearModels
end

# ╔═╡ 97101f45-c447-4fd1-bbfd-d4fead047118
md"
This notebook shows how we dealt with our data for the project. There are some plots helping visualize the data and how we dealt with missing data.
"

# ╔═╡ fbd4c00b-9ffe-4656-a98b-09cc80bf4e58
begin
	training_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
	test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
	
	coerce!(training_data, :precipitation_nextday => Multiclass);
	coerce!(test_data, :precipitation_nextday => Multiclass);
end

# ╔═╡ 0d8b1166-b9c8-44ae-abfa-9060e9a7dd33
md"
We can see that out training data consists of 3176 rows and 529 columns. We have 528 parameters in order to predict the 529 column, wether or not it is raining the next day in Pully called :precipitation_nextday
"

# ╔═╡ cb8b6542-1ec9-41a2-8508-bb7acc944e32
size(training_data)

# ╔═╡ 1b27bf59-2f01-4802-bac9-c325ba1681c3
describe(training_data)

# ╔═╡ c31e9d13-c4fe-4350-9fdf-3ca7450799d9
md"
Droppring the missing data gives us only 1699 data points left.
"

# ╔═╡ dd4ce6ae-2618-4c07-90b6-19c4eea00a37
begin
	drop = dropmissing(training_data)
	coerce!(drop, :precipitation_nextday => Multiclass);
	size(drop)
end

# ╔═╡ d0a77735-69e1-4093-b153-796b395b82b0
md"
Whereas filling the missing data allow us to keep the the 3176 original points.
"

# ╔═╡ 011a56a8-03a5-4580-b79e-8d1feb69e3f7
begin
	data_med = MLJ.transform(fit!(machine(FillImputer(), select(training_data[:,:], Not([:precipitation_nextday])))), select(training_data[:,:], Not([:precipitation_nextday])));
    insertcols!(data_med,:precipitation_nextday=>training_data.precipitation_nextday[:]);
    coerce!(data_med, :precipitation_nextday => Multiclass);
	size(data_med)
end

# ╔═╡ 76adc00d-081e-4cd0-92a6-07d2d7599fb3
md"
Lookin at PUY meteo measures in order to observe first result correlation in our data.
"

# ╔═╡ 31286f5d-33cb-4399-9744-139655a4e919
@df drop corrplot([:PUY_radiation_1 :PUY_delta_pressure_1 :PUY_air_temp_1 :PUY_wind_1],
                     grid = false, fillcolor = cgrad(), size = (1000, 1000))

# ╔═╡ 09bd5a64-48fa-45fb-9323-7205d7e3ab3b
md"
Trouver un moyen de visualizer histogram avec true et false maybe ?
"

# ╔═╡ 116abdc3-87d4-4bc3-b35d-531e14fec93b
histogram(training_data.PUY_sunshine_4, training_data.precipitation_nextday,xlabel="PUY_air_temp_4", ylabel="precipitation_nextday")

# ╔═╡ 90133410-7311-45a0-8962-ad293b3f6e14
md"
Performing PCA on our data set to access the most import features explaining the variances in :precipitation_nextday
"

# ╔═╡ 858f6ae0-caf1-4cae-9ec3-0333f2be2f0b
md"
By looking a the result in a terminal thanks to the verbosity argument it is possible to see that 2 predictors have mean and std equal to zero; ZER_sunshine_1 and ALT_sunshine_4 we need to drop these 2 columns in order to standardize
"

# ╔═╡ cc06a386-7edb-42fa-a4e2-fa43b2d79779
begin
	
	mach_drop_std = machine(Standardizer(), select(drop, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
	fit!(mach_drop_std, verbosity = 2)
	data_drop_std = MLJ.transform(mach_drop_std, select(drop, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
end

# ╔═╡ 4224387b-155f-4bff-944d-4254d54e01a7
size(data_drop_std)

# ╔═╡ bada701a-79eb-40a0-959d-b5ae148c55a9
begin
	mach_med_std = machine(Standardizer(), select(data_med, Not([:ZER_sunshine_1, :ALT_sunshine_4]))
)
	fit!(mach_med_std)
	data_med_std = MLJ.transform(mach_med_std, select(data_med, Not([:ZER_sunshine_1, :ALT_sunshine_4]))
)
end

# ╔═╡ 24682b9b-505f-4beb-ae32-eafc58449d89
size(data_med_std)

# ╔═╡ 89a6c6a2-0c2f-4711-af05-7a905cdb57f4
pca_med = fit!(machine(PCA(pratio = 1), data_med_std));

# ╔═╡ 1b03e1d8-8e90-4796-913a-3c5864ec093a
pca_drop = fit!(machine(PCA(pratio = 1), data_drop_std));

# ╔═╡ b63d208f-915b-45db-963b-f7f4ef23a2eb
report(pca_med)

# ╔═╡ 95589fe4-42e7-4936-9981-78845cccb1ce
report(pca_drop)

# ╔═╡ a1a8a212-b50f-494e-bb63-79e5544c9267
let
	gr()
	biplot(pca_med)
end

# ╔═╡ 14525726-dd52-4ec1-a68b-51d0cd99ef7a
let
	gr()
	biplot(pca_drop)
end

# ╔═╡ Cell order:
# ╟─97101f45-c447-4fd1-bbfd-d4fead047118
# ╠═b0ded90e-6283-11ec-0039-2de048031f30
# ╠═fbd4c00b-9ffe-4656-a98b-09cc80bf4e58
# ╠═0d8b1166-b9c8-44ae-abfa-9060e9a7dd33
# ╠═cb8b6542-1ec9-41a2-8508-bb7acc944e32
# ╠═1b27bf59-2f01-4802-bac9-c325ba1681c3
# ╠═c31e9d13-c4fe-4350-9fdf-3ca7450799d9
# ╠═dd4ce6ae-2618-4c07-90b6-19c4eea00a37
# ╠═d0a77735-69e1-4093-b153-796b395b82b0
# ╠═011a56a8-03a5-4580-b79e-8d1feb69e3f7
# ╠═76adc00d-081e-4cd0-92a6-07d2d7599fb3
# ╠═31286f5d-33cb-4399-9744-139655a4e919
# ╠═09bd5a64-48fa-45fb-9323-7205d7e3ab3b
# ╠═116abdc3-87d4-4bc3-b35d-531e14fec93b
# ╠═90133410-7311-45a0-8962-ad293b3f6e14
# ╠═858f6ae0-caf1-4cae-9ec3-0333f2be2f0b
# ╠═cc06a386-7edb-42fa-a4e2-fa43b2d79779
# ╠═4224387b-155f-4bff-944d-4254d54e01a7
# ╠═bada701a-79eb-40a0-959d-b5ae148c55a9
# ╠═24682b9b-505f-4beb-ae32-eafc58449d89
# ╠═89a6c6a2-0c2f-4711-af05-7a905cdb57f4
# ╠═1b03e1d8-8e90-4796-913a-3c5864ec093a
# ╠═b63d208f-915b-45db-963b-f7f4ef23a2eb
# ╠═95589fe4-42e7-4936-9981-78845cccb1ce
# ╠═a1a8a212-b50f-494e-bb63-79e5544c9267
# ╠═14525726-dd52-4ec1-a68b-51d0cd99ef7a
