### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 1c2aaf33-957c-4c31-8a48-428cab2e738e
begin
	using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	using Plots, OpenML, CSV, StatsPlots, MLJ, DataFrames, MLJLinearModels, Random
end

# ╔═╡ 71230022-aecc-4c49-8b37-145a95372474
using NearestNeighborModels

# ╔═╡ 53920a34-2b9c-4003-a5a0-d0fd444d1ea3
weather = CSV.read(joinpath(@__DIR__, "data","trainingdata.csv"), DataFrame)

# ╔═╡ b682458f-dd14-4d3a-a88d-228991a29d9d
test = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame);

# ╔═╡ 4d9e886f-8841-49af-b60c-a5ff11fd7319
example_submission = CSV.read(joinpath(@__DIR__, "data", "sample_submission.csv"), DataFrame);

# ╔═╡ f8f72a97-e8a4-43f8-bbcc-3f52b102d18e
size(weather)

# ╔═╡ 459d46cf-cd78-49e6-bd24-26ff84aa17c8
describe(weather)

# ╔═╡ 3410d4cf-7812-4092-8024-68d080ab37a1
begin
	weather_no_missing = dropmissing(weather)
	size(weather_no_missing)
end

# ╔═╡ a7b563b5-a3b4-4bc7-afa5-ae167d53b3bd
md"""
Here it is not a good idea to use the function dropmissing (removes rows with missing values) as it's removing almost half of the data !

We use FillIputer, which imputes missing data with a fixed value computed on the non-missing values. --> Try to change which value it computesßSß
"""

# ╔═╡ 8cd10d8d-a608-44f6-a92d-e1814309fa01
imputed_data = MLJ.transform(fit!(machine(FillImputer(), select(weather[:,:], Not([:precipitation_nextday])))), select(weather[:,:], Not([:precipitation_nextday])));

# ╔═╡ 5cf1cf4f-fe14-4fde-a6c6-eea708767e02
size(imputed_data)

# ╔═╡ d8ab190d-bd5c-4635-902f-efcac1afa91f
data = insertcols!(imputed_data,:precipitation_nextday=>weather.precipitation_nextday[:]);

# ╔═╡ d473ba62-1717-4953-9d47-cbcee758115c
coerce!(data, :precipitation_nextday => Multiclass);

# ╔═╡ a50d5a73-f0fe-41db-8b68-ebbc4e223360
mach1 = machine(LogisticClassifier(penalty = :l2, lambda = 2e-2),
                        select(data[1:2500,:], Not(:precipitation_nextday)),
                        data.precipitation_nextday[1:2500]) |> fit!;

# ╔═╡ 11a7232f-7090-45d3-bad8-9dec70350faa
predict(mach1, data[2501:3176,:])

# ╔═╡ b7abd5cb-e0c2-4e30-bb82-c6f8552e081a
predict_mode(mach1, data[2501:3176,:])

# ╔═╡ cdc2a831-d896-4ea4-8e66-5620cf161bc4
function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response)
	)
end;

# ╔═╡ 9f4ca23b-3d43-4c2a-b741-0cc97f5bd9af
losses(mach1, data[1:2500,:], data.precipitation_nextday[1:2500])

# ╔═╡ ec51b7fe-3230-4d9f-98be-2a8868db6523
losses(mach1, data[2501:3176,:], data.precipitation_nextday[2501:3176])

# ╔═╡ 9f09b178-44d3-41d0-8733-cfa25fdc8c8a
confusion_matrix(predict_mode(mach1, select(data[1:2500,:],
	Not(:precipitation_nextday))),
	data.precipitation_nextday[1:2500])

# ╔═╡ 89ccf6e1-030a-4a9d-a26d-66d3a62c3e67
confusion_matrix(predict_mode(mach1, select(data[2501:3176,:],
	Not(:precipitation_nextday))),
	data.precipitation_nextday[2501:3176])

# ╔═╡ d2b336a3-ae27-4d28-a61f-b32ccb15ba45
md"
## K Nearest Neighbors
"

# ╔═╡ b76cbf48-8dec-4a92-b439-8fef55014af9
machine_KNN = machine(KNNClassifier(K = 36), select(data[1:2500,:],
	Not(:precipitation_nextday)), data.precipitation_nextday[1:2500]) |>fit!

# ╔═╡ 6e6369a7-2d11-48c1-a2cc-39a292d35891
err_train = mean(predict_mode(machine_KNN, select(data[1:2500,:],
	Not(:precipitation_nextday))) .!= data.precipitation_nextday[1:2500])

# ╔═╡ 46f87d0d-dc29-4cdf-92cd-21a768c50e2a
err = mean(predict_mode(machine_KNN, select(data[2501:3176,:],
	Not(:precipitation_nextday))) .!= data.precipitation_nextday[2501:3176])

# ╔═╡ 4df67452-8b99-4403-85a1-46adf955deb2
auc_KNN_train = MLJ.auc(predict(machine_KNN, select(data[1:2500,:],
	Not(:precipitation_nextday))), data.precipitation_nextday[1:2500])

# ╔═╡ 9eef9adc-7b3e-49cb-a469-d0ac2fcf2046
auc_KNN = MLJ.auc(predict(machine_KNN, select(data[2501:3176,:],
	Not(:precipitation_nextday))), data.precipitation_nextday[2501:3176])

# ╔═╡ 6933ec43-6260-4b10-86e0-35e77b4d16f8
pred = pdf.(predict(machine_KNN, test), true)

# ╔═╡ 0fadd7ed-e7ea-45b7-952d-af3090b88dc3
example_submission.precipitation_nextday = pred

# ╔═╡ e5ce448a-eb18-4834-bc4c-33e5e1ba575a
CSV.write(joinpath(@__DIR__,"..","submission.csv"), example_submission)

# ╔═╡ 76fd15c6-646f-4d4e-b22a-b8a8ab14ec28
begin
    model_KNN = KNNClassifier()
    self_tuning_model = TunedModel(model = model_KNN,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(),
                                   range = range(model_KNN, :K, values = 1:50),
                                   measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                               select(data, Not(:precipitation_nextday)),
                               data.precipitation_nextday) |> fit!
end

# ╔═╡ d739fdf1-ad0c-41a4-8785-590417ab86a1
rep = report(self_tuning_mach)

# ╔═╡ cfadd398-3c83-4aa2-be44-2f4dcc709202
scatter(reshape(rep.plotting.parameter_values, :),
	    rep.plotting.measurements, xlabel = "K", ylabel = "AUC")

# ╔═╡ Cell order:
# ╠═1c2aaf33-957c-4c31-8a48-428cab2e738e
# ╠═53920a34-2b9c-4003-a5a0-d0fd444d1ea3
# ╠═b682458f-dd14-4d3a-a88d-228991a29d9d
# ╠═4d9e886f-8841-49af-b60c-a5ff11fd7319
# ╠═f8f72a97-e8a4-43f8-bbcc-3f52b102d18e
# ╠═459d46cf-cd78-49e6-bd24-26ff84aa17c8
# ╠═3410d4cf-7812-4092-8024-68d080ab37a1
# ╟─a7b563b5-a3b4-4bc7-afa5-ae167d53b3bd
# ╠═8cd10d8d-a608-44f6-a92d-e1814309fa01
# ╠═5cf1cf4f-fe14-4fde-a6c6-eea708767e02
# ╠═d8ab190d-bd5c-4635-902f-efcac1afa91f
# ╠═d473ba62-1717-4953-9d47-cbcee758115c
# ╠═a50d5a73-f0fe-41db-8b68-ebbc4e223360
# ╠═11a7232f-7090-45d3-bad8-9dec70350faa
# ╠═b7abd5cb-e0c2-4e30-bb82-c6f8552e081a
# ╠═cdc2a831-d896-4ea4-8e66-5620cf161bc4
# ╠═9f4ca23b-3d43-4c2a-b741-0cc97f5bd9af
# ╠═ec51b7fe-3230-4d9f-98be-2a8868db6523
# ╠═9f09b178-44d3-41d0-8733-cfa25fdc8c8a
# ╠═89ccf6e1-030a-4a9d-a26d-66d3a62c3e67
# ╟─d2b336a3-ae27-4d28-a61f-b32ccb15ba45
# ╠═71230022-aecc-4c49-8b37-145a95372474
# ╠═b76cbf48-8dec-4a92-b439-8fef55014af9
# ╠═6e6369a7-2d11-48c1-a2cc-39a292d35891
# ╠═46f87d0d-dc29-4cdf-92cd-21a768c50e2a
# ╠═4df67452-8b99-4403-85a1-46adf955deb2
# ╠═9eef9adc-7b3e-49cb-a469-d0ac2fcf2046
# ╠═6933ec43-6260-4b10-86e0-35e77b4d16f8
# ╠═0fadd7ed-e7ea-45b7-952d-af3090b88dc3
# ╠═e5ce448a-eb18-4834-bc4c-33e5e1ba575a
# ╠═76fd15c6-646f-4d4e-b22a-b8a8ab14ec28
# ╠═d739fdf1-ad0c-41a4-8785-590417ab86a1
# ╠═cfadd398-3c83-4aa2-be44-2f4dcc709202
