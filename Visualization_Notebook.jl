### A Pluto.jl notebook ###
# v0.17.3

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

# ╔═╡ 6bf0c668-d6e4-474e-aef3-79e5fcde5538
md"
## Loading the packages
"

# ╔═╡ 1f8ded80-aee0-47ee-9617-b6cc53b6c51a
md"
## First look at the data sets
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
We can use the function describe(::DataFrames) in order to assess the first properties of each predictor. This can help us to find the predictor with mean and std equal to 0 which will be a problem when standardizing.
"

# ╔═╡ 1b27bf59-2f01-4802-bac9-c325ba1681c3
describe(training_data)

# ╔═╡ 81fb7d32-24dc-460b-85df-aecdd8263e91
md"
## Dealing with missing data
"

# ╔═╡ 406831e5-e18a-40ba-a0da-936774d48425
md"
Missing data points is one of the first step in ML project, because the models will need complete data set to be fitted on.
"

# ╔═╡ c31e9d13-c4fe-4350-9fdf-3ca7450799d9
md"
When dropping the data points containing missing data we are left with 1699 data points.
"

# ╔═╡ dd4ce6ae-2618-4c07-90b6-19c4eea00a37
begin
	data_drop = dropmissing(training_data)
	coerce!(data_drop, :precipitation_nextday => Multiclass);
	size(data_drop)
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

# ╔═╡ cbfa4870-8db1-4e5d-ae0e-27d1c2b55066
md"
Is there any good reason to put the median in those empty cells ? It is probably better than putting complete random data, but other ways of filling these cells can be studied.
It still allow us to concerve all the data points and thus a lot of information from the original data set.
"

# ╔═╡ 4593a912-1a19-4e65-bdc5-a777681bbcf6
md"
## Visualization
"

# ╔═╡ 76adc00d-081e-4cd0-92a6-07d2d7599fb3
md"
Lookin at Pully measurements in order to observe first result correlation in our data.
"

# ╔═╡ 31286f5d-33cb-4399-9744-139655a4e919
@df data_drop corrplot([:PUY_radiation_1 :PUY_delta_pressure_1 :PUY_air_temp_1 :PUY_wind_1], grid = false, fillcolor = cgrad(), size = (1000, 1000))

# ╔═╡ 1f6ce02d-40c5-43e4-a8a6-889fae7fbd77
md"
## Standardization
"

# ╔═╡ 46da38df-4c18-4767-8325-db55454487e6
md"
In order to standardize our data, we need to find the predictors that have a mean and a std equal to zero. We can use the fit!(..., verbosity=2) function on the Standardizer() machine. This will give us in our terminal which predictors have a mean and std equal to 0. Having the mean and std equal to 0 lead to the fraction 0/0 during the standardization, thus creating NaN cells which are a problem when fitting models." 

# ╔═╡ e1b4eb7e-f6a3-4510-908d-922b3211f300
md"
In this step we found that :ZER\_sunshine\_1 and :ALT\_sunshine\_4 have a mean and a std equal to zero, we drop these two columns in order to standardize the data."

# ╔═╡ cc06a386-7edb-42fa-a4e2-fa43b2d79779
begin
	mach_drop_std = machine(Standardizer(), select(data_drop, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
	fit!(mach_drop_std, verbosity = 2)
	data_drop_std = MLJ.transform(mach_drop_std, select(data_drop, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
end

# ╔═╡ 40430b02-1a70-45fd-9f14-d446c95f77eb
md"
When we look at the size of the data set, we can see that it has 2 predictors less now, so 526 predictors and 1 ouput columns."

# ╔═╡ 4224387b-155f-4bff-944d-4254d54e01a7
size(data_drop_std)

# ╔═╡ 419eb119-613e-431c-8126-a408027084e0
md"
Same thing for the filled out data. "

# ╔═╡ bada701a-79eb-40a0-959d-b5ae148c55a9
begin
	mach_med_std = machine(Standardizer(), select(data_med, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
	fit!(mach_med_std)
	data_med_std = MLJ.transform(mach_med_std, select(data_med, Not([:ZER_sunshine_1, :ALT_sunshine_4]))
)
end

# ╔═╡ 24682b9b-505f-4beb-ae32-eafc58449d89
size(data_med_std)

# ╔═╡ 09682503-f59e-4c84-9bcb-bab37806b71f
md"
## Principal Component Analysis (PCA)
"

# ╔═╡ 90133410-7311-45a0-8962-ad293b3f6e14
md"
We perform PCA on our data set to assess the principal features that explain the variance in :precipitation_nextday.
"

# ╔═╡ 1b03e1d8-8e90-4796-913a-3c5864ec093a
pca_drop = fit!(machine(PCA(pratio = 1), data_drop_std));

# ╔═╡ 89a6c6a2-0c2f-4711-af05-7a905cdb57f4
pca_med = fit!(machine(PCA(pratio = 1), data_med_std));

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

# ╔═╡ 267b717b-cc3e-4b1a-ac2c-416ffd9379df
md"
This first introduction to the data sets will allow us to tuned our model more precisely, a important function called generate(; option, std, valid, test) allow the transformation of the data in the file Data\_processing.jl that will be used in every ML methods"

# ╔═╡ Cell order:
# ╟─97101f45-c447-4fd1-bbfd-d4fead047118
# ╟─6bf0c668-d6e4-474e-aef3-79e5fcde5538
# ╠═b0ded90e-6283-11ec-0039-2de048031f30
# ╟─1f8ded80-aee0-47ee-9617-b6cc53b6c51a
# ╟─75c3f94d-8d0a-4f6f-ab50-41e46541c3d8
# ╠═fbd4c00b-9ffe-4656-a98b-09cc80bf4e58
# ╠═29b461df-effd-4e0d-8d48-56e63c07913d
# ╟─0d8b1166-b9c8-44ae-abfa-9060e9a7dd33
# ╠═cb8b6542-1ec9-41a2-8508-bb7acc944e32
# ╠═ef87c1fb-eaa1-459b-b610-737259a34586
# ╟─ebdf399c-72f4-4aee-92b2-08727423d174
# ╠═1b27bf59-2f01-4802-bac9-c325ba1681c3
# ╟─81fb7d32-24dc-460b-85df-aecdd8263e91
# ╟─406831e5-e18a-40ba-a0da-936774d48425
# ╟─c31e9d13-c4fe-4350-9fdf-3ca7450799d9
# ╠═dd4ce6ae-2618-4c07-90b6-19c4eea00a37
# ╟─d0a77735-69e1-4093-b153-796b395b82b0
# ╠═011a56a8-03a5-4580-b79e-8d1feb69e3f7
# ╟─cbfa4870-8db1-4e5d-ae0e-27d1c2b55066
# ╟─4593a912-1a19-4e65-bdc5-a777681bbcf6
# ╟─76adc00d-081e-4cd0-92a6-07d2d7599fb3
# ╠═31286f5d-33cb-4399-9744-139655a4e919
# ╟─1f6ce02d-40c5-43e4-a8a6-889fae7fbd77
# ╟─46da38df-4c18-4767-8325-db55454487e6
# ╟─e1b4eb7e-f6a3-4510-908d-922b3211f300
# ╠═cc06a386-7edb-42fa-a4e2-fa43b2d79779
# ╟─40430b02-1a70-45fd-9f14-d446c95f77eb
# ╠═4224387b-155f-4bff-944d-4254d54e01a7
# ╟─419eb119-613e-431c-8126-a408027084e0
# ╠═bada701a-79eb-40a0-959d-b5ae148c55a9
# ╠═24682b9b-505f-4beb-ae32-eafc58449d89
# ╟─09682503-f59e-4c84-9bcb-bab37806b71f
# ╟─90133410-7311-45a0-8962-ad293b3f6e14
# ╠═1b03e1d8-8e90-4796-913a-3c5864ec093a
# ╠═89a6c6a2-0c2f-4711-af05-7a905cdb57f4
# ╠═b63d208f-915b-45db-963b-f7f4ef23a2eb
# ╠═95589fe4-42e7-4936-9981-78845cccb1ce
# ╠═a1a8a212-b50f-494e-bb63-79e5544c9267
# ╠═14525726-dd52-4ec1-a68b-51d0cd99ef7a
# ╟─9861d8a5-dce3-4446-a648-357829bb63a1
# ╠═886ac080-398e-4060-b419-28b636e77bac
# ╟─9cb38d7b-6065-4756-acfb-f3e888539fdd
# ╠═5af13796-7a27-4cc5-8cc1-90679500f0ac
# ╟─267b717b-cc3e-4b1a-ac2c-416ffd9379df
