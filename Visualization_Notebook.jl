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
# Will it rain tomorrow ?
This notebook shows how we dealt with the training data set given for the project [Will it rain tomorrow ?](https://www.kaggle.com/c/bio322-will-it-rain-tomorrow).
We propose some plots of the data and the transformation we used in our models.

First we need to load the necessary packages. We used the MLCourse env in order to access the packages.
"

# ╔═╡ 75c3f94d-8d0a-4f6f-ab50-41e46541c3d8
md"
Then we load the data sets;
1. The training data set
2. The test data set
"

# ╔═╡ fbd4c00b-9ffe-4656-a98b-09cc80bf4e58
begin
	training_data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
	coerce!(training_data, :precipitation_nextday => Multiclass);
end

# ╔═╡ 29b461df-effd-4e0d-8d48-56e63c07913d
begin
	test_data = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);
end

# ╔═╡ 19d596c6-3101-41ff-bc21-510a69d8e07d


# ╔═╡ 0d8b1166-b9c8-44ae-abfa-9060e9a7dd33
md"
We can see that out training data consists of 3176 rows and 529 columns. We have 528 predictors in order to predict the 529th column, whether or not it is raining the next day in Pully called :precipitation\_nextday.
The test is then composed only by the 528 predictors with different data points. We can then use this data set top predict the :precipitation\_nextday output using our models.
"

# ╔═╡ cb8b6542-1ec9-41a2-8508-bb7acc944e32
size(training_data)

# ╔═╡ ef87c1fb-eaa1-459b-b610-737259a34586
size(test_data)

# ╔═╡ ebdf399c-72f4-4aee-92b2-08727423d174
md"
We can use the function describe(::DataFrames) in order to assess the first properties of each predictor. This can help us to find the predictor with mean and std equal to 0 which will be a problem when standardizing
"

# ╔═╡ 1b27bf59-2f01-4802-bac9-c325ba1681c3
describe(training_data)

# ╔═╡ 81fb7d32-24dc-460b-85df-aecdd8263e91
md"
## Dealing with missing data
"

# ╔═╡ c31e9d13-c4fe-4350-9fdf-3ca7450799d9
md"
When dropping the data points containing missing data we are left with 1699 data points.
"

# ╔═╡ dd4ce6ae-2618-4c07-90b6-19c4eea00a37
begin
	drop = dropmissing(training_data)
	coerce!(drop, :precipitation_nextday => Multiclass);
	size(drop)
end

# ╔═╡ d0a77735-69e1-4093-b153-796b395b82b0
md"
Whereas filling the missing data allows us to keep the the 3176 original points.
"

# ╔═╡ 011a56a8-03a5-4580-b79e-8d1feb69e3f7
begin
	data_med = MLJ.transform(fit!(machine(FillImputer(), select(training_data[:,:], Not([:precipitation_nextday])))), select(training_data[:,:], Not([:precipitation_nextday])));
    insertcols!(data_med,:precipitation_nextday=>training_data.precipitation_nextday[:]);
    coerce!(data_med, :precipitation_nextday => Multiclass);
	size(data_med)
end

# ╔═╡ 4593a912-1a19-4e65-bdc5-a777681bbcf6
md"
## Visualization
"

# ╔═╡ 76adc00d-081e-4cd0-92a6-07d2d7599fb3
md"
Lookin at Pully measurements in order to observe first result correlation in our data.
"

# ╔═╡ 31286f5d-33cb-4399-9744-139655a4e919
@df drop corrplot([:PUY_radiation_1 :PUY_delta_pressure_1 :PUY_air_temp_1 :PUY_wind_1],
                     grid = false, fillcolor = cgrad(), size = (1000, 1000))

# ╔═╡ 09682503-f59e-4c84-9bcb-bab37806b71f
md"
## PCA
"

# ╔═╡ 90133410-7311-45a0-8962-ad293b3f6e14
md"
Performing PCA on our data set to access the most import features explaining the variances in :precipitation_nextday:
"

# ╔═╡ 858f6ae0-caf1-4cae-9ec3-0333f2be2f0b
md"
By looking a the result in a terminal thanks to the verbosity argument it is possible to see that 2 predictors have mean and standard deviation equal to zero: ZER_sunshine_1 and ALT_sunshine_4. We need to drop these 2 columns in order to standardize our data.
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
	mach_med_std = machine(Standardizer(), select(data_med, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
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

# ╔═╡ 9861d8a5-dce3-4446-a648-357829bb63a1
md"
Below we can see the proportion and cumulative proportion of variance explained using the dataset with median-imputed missing data.
"

# ╔═╡ 886ac080-398e-4060-b419-28b636e77bac
let
    vars = report(pca_med).principalvars ./ report(pca_med).tvar
    p1 = plot(vars, label = nothing, yscale = :log10,
              xlabel = "component", ylabel = "proportion of variance explained")
    p2 = plot(cumsum(vars),
              label = nothing, xlabel = "component",
              ylabel = "cumulative prop. of variance explained")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end

# ╔═╡ 9cb38d7b-6065-4756-acfb-f3e888539fdd
md"
Below we can see the proportion and cumulative proportion of variance explained using the dataset with the data points containing missing data removed.
"

# ╔═╡ 5af13796-7a27-4cc5-8cc1-90679500f0ac
let
    vars = report(pca_drop).principalvars ./ report(pca_drop).tvar
    p1 = plot(vars, label = nothing, yscale = :log10,
              xlabel = "component", ylabel = "proportion of variance explained")
    p2 = plot(cumsum(vars),
              label = nothing, xlabel = "component",
              ylabel = "cumulative prop. of variance explained")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end

# ╔═╡ Cell order:
# ╟─97101f45-c447-4fd1-bbfd-d4fead047118
# ╠═b0ded90e-6283-11ec-0039-2de048031f30
# ╟─75c3f94d-8d0a-4f6f-ab50-41e46541c3d8
# ╠═fbd4c00b-9ffe-4656-a98b-09cc80bf4e58
# ╠═29b461df-effd-4e0d-8d48-56e63c07913d
# ╠═19d596c6-3101-41ff-bc21-510a69d8e07d
# ╟─0d8b1166-b9c8-44ae-abfa-9060e9a7dd33
# ╠═cb8b6542-1ec9-41a2-8508-bb7acc944e32
# ╠═ef87c1fb-eaa1-459b-b610-737259a34586
# ╟─ebdf399c-72f4-4aee-92b2-08727423d174
# ╠═1b27bf59-2f01-4802-bac9-c325ba1681c3
# ╟─81fb7d32-24dc-460b-85df-aecdd8263e91
# ╟─c31e9d13-c4fe-4350-9fdf-3ca7450799d9
# ╠═dd4ce6ae-2618-4c07-90b6-19c4eea00a37
# ╟─d0a77735-69e1-4093-b153-796b395b82b0
# ╠═011a56a8-03a5-4580-b79e-8d1feb69e3f7
# ╟─4593a912-1a19-4e65-bdc5-a777681bbcf6
# ╟─76adc00d-081e-4cd0-92a6-07d2d7599fb3
# ╠═31286f5d-33cb-4399-9744-139655a4e919
# ╟─09682503-f59e-4c84-9bcb-bab37806b71f
# ╟─90133410-7311-45a0-8962-ad293b3f6e14
# ╟─858f6ae0-caf1-4cae-9ec3-0333f2be2f0b
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
# ╟─9861d8a5-dce3-4446-a648-357829bb63a1
# ╟─886ac080-398e-4060-b419-28b636e77bac
# ╟─9cb38d7b-6065-4756-acfb-f3e888539fdd
# ╟─5af13796-7a27-4cc5-8cc1-90679500f0ac
