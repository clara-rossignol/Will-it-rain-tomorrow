### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ ac8c0d24-580e-11ec-086e-f9c74660381a
begin
	using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	import Pkg; Pkg.add("MLJModels")
	using Plots, OpenML, CSV, StatsPlots, MLJ, DataFrames, MLJLinearModels, Random
end

# ╔═╡ 53920a34-2b9c-4003-a5a0-d0fd444d1ea3
weather = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame)

# ╔═╡ f8f72a97-e8a4-43f8-bbcc-3f52b102d18e
size(weather)

# ╔═╡ 459d46cf-cd78-49e6-bd24-26ff84aa17c8
describe(weather)

# ╔═╡ 3410d4cf-7812-4092-8024-68d080ab37a1
weather_no_missing = dropmissing(weather)

# ╔═╡ aec3947b-fdfc-4c08-825d-82657f5c4ecb
size(weather_no_missing)

# ╔═╡ a7b563b5-a3b4-4bc7-afa5-ae167d53b3bd
md"""
Maybe not the best to use dropmissing as it's removing almost half of the data !
Replace values with mean or median or other?
"""

# ╔═╡ 8cd10d8d-a608-44f6-a92d-e1814309fa01
imputed_data = MLJ.transform(fit!(machine(FillImputer(), select(weather[:,:], Not([:precipitation_nextday])))), select(weather[:,:], Not([:precipitation_nextday])))

# ╔═╡ 5cf1cf4f-fe14-4fde-a6c6-eea708767e02
size(imputed_data)

# ╔═╡ 54b3d5e4-f4f2-46d7-a2e2-b21ed9669c8b
output = weather.precipitation_nextday[:]

# ╔═╡ 66634826-2b9a-4e16-8a7a-1abe65705dc5
data = insertcols!(imputed_data,:precipitation_nextday=>output)

# ╔═╡ c942200e-b63b-4332-a0ca-061718eb2873
y = data.CHU_delta_pressure_1[:]

# ╔═╡ aa983fe5-395e-4744-93f8-77c9825ca4fa
X = data.GVE_delta_pressure_1[:]

# ╔═╡ 7212b2c4-eb2b-40dc-b004-eeea145ed5fe
histogram2d(X, y, markersize = 3, xlabel = "GVE_delta_pressure_1",
	        legend = false, ylabel = "CHU_delta_pressure_1", bins = (250, 200))

# ╔═╡ 819fe9bc-662f-4d92-a037-22e80593cf32
md"
standardize input ? 
fine tuning : use ISTA, FISTA or ProxGrad with L1 regularization (each with its meta-parameters).
"

# ╔═╡ c6f44418-85ad-4c56-be1b-c0194fb3ba0b
mach1 = machine(LogisticClassifier(penalty = :l2, lambda = 2e-2),
                        select(data[1:3000,:], Not(:precipitation_nextday)),
                        data.precipitation_nextday[1:3000]) |> fit!;

# ╔═╡ 11a7232f-7090-45d3-bad8-9dec70350faa
predict(mach1, data[3001:3176,:])

# ╔═╡ b7abd5cb-e0c2-4e30-bb82-c6f8552e081a
predict_mode(mach1, data[3001:3176,:])

# ╔═╡ cdc2a831-d896-4ea4-8e66-5620cf161bc4
function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response)
	)
end;

# ╔═╡ 9f4ca23b-3d43-4c2a-b741-0cc97f5bd9af
losses(mach1, data[1:3000,:], data.precipitation_nextday[1:3000])

# ╔═╡ ec51b7fe-3230-4d9f-98be-2a8868db6523
losses(mach1, data[3001:3176,:], data.precipitation_nextday[3001:3176])

# ╔═╡ 9f09b178-44d3-41d0-8733-cfa25fdc8c8a
confusion_matrix(predict_mode(mach1, select(data[1:3000,:],
	Not(:precipitation_nextday))),
	data.precipitation_nextday[1:3000])

# ╔═╡ 89ccf6e1-030a-4a9d-a26d-66d3a62c3e67
confusion_matrix(predict_mode(mach1, select(data[3001:3176,:],
	Not(:precipitation_nextday))),
	data.precipitation_nextday[3001:3176])

# ╔═╡ Cell order:
# ╠═ac8c0d24-580e-11ec-086e-f9c74660381a
# ╠═53920a34-2b9c-4003-a5a0-d0fd444d1ea3
# ╠═f8f72a97-e8a4-43f8-bbcc-3f52b102d18e
# ╠═459d46cf-cd78-49e6-bd24-26ff84aa17c8
# ╠═3410d4cf-7812-4092-8024-68d080ab37a1
# ╠═aec3947b-fdfc-4c08-825d-82657f5c4ecb
# ╟─a7b563b5-a3b4-4bc7-afa5-ae167d53b3bd
# ╠═8cd10d8d-a608-44f6-a92d-e1814309fa01
# ╠═5cf1cf4f-fe14-4fde-a6c6-eea708767e02
# ╠═54b3d5e4-f4f2-46d7-a2e2-b21ed9669c8b
# ╠═66634826-2b9a-4e16-8a7a-1abe65705dc5
# ╠═c942200e-b63b-4332-a0ca-061718eb2873
# ╠═aa983fe5-395e-4744-93f8-77c9825ca4fa
# ╠═7212b2c4-eb2b-40dc-b004-eeea145ed5fe
# ╟─819fe9bc-662f-4d92-a037-22e80593cf32
# ╠═c6f44418-85ad-4c56-be1b-c0194fb3ba0b
# ╠═11a7232f-7090-45d3-bad8-9dec70350faa
# ╠═b7abd5cb-e0c2-4e30-bb82-c6f8552e081a
# ╠═cdc2a831-d896-4ea4-8e66-5620cf161bc4
# ╠═9f4ca23b-3d43-4c2a-b741-0cc97f5bd9af
# ╠═ec51b7fe-3230-4d9f-98be-2a8868db6523
# ╠═9f09b178-44d3-41d0-8733-cfa25fdc8c8a
# ╠═89ccf6e1-030a-4a9d-a26d-66d3a62c3e67
