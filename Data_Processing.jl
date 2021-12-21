using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, MLJ, DataFrames

function generate(data, valid = "true")
    data_drop = dropmissing(data);
    data_med = MLJ.transform(fit!(machine(FillImputer(), select(data[:,:], Not([:precipitation_nextday])))), select(data[:,:], Not([:precipitation_nextday])));
    insertcols!(data_med,:precipitation_nextday=>data.precipitation_nextday[:]);

    coerce!(data_drop, :precipitation_nextday => Multiclass);
    coerce!(data_med, :precipitation_nextday => Multiclass);

    drop_mach = fit!(machine(Standardizer(), select(data_drop[:,:], Not(:precipitation_nextday))));
    med_mach = fit!(machine(Standardizer(), select(data_med[:,:], Not(:precipitation_nextday))));
    data_drop_std = MLJ.transform(drop_mach, select(data_drop[:,:], Not(:precipitation_nextday)));
    data_med_std = MLJ.transform(med_mach, select(data_med[:,:], Not(:precipitation_nextday)));
    insertcols!(data_drop_std,:precipitation_nextday=>data_drop.precipitation_nextday[:]);
    insertcols!(data_med_std,:precipitation_nextday=>data_med.precipitation_nextday[:]);

    coerce!(data_drop_std, :precipitation_nextday => Multiclass);
    coerce!(data_med_std, :precipitation_nextday => Multiclass);

    coerce!(data_drop_std, Count => MLJ.Continuous)
    coerce!(data_med_std, Count => MLJ.Continuous)


    Random.seed!(2809)
    idxs1 = randperm(size(data_drop, 1));
    idxs2 = randperm(size(data_med, 1));
    dim_drop = size(data_drop)[1];
    dim_med = size(data_med)[1];

    idx_drop = Vector{Int}([1, floor(dim_drop/3), floor(2*dim_drop/3), dim_drop]);
    #1 / 169 / 339 / 1699
    idx_med = Vector{Int}([1, floor(dim_med/3), floor(2*dim_med/3), dim_med]);
    #1 / 317 / 653 / 3176
    
    if valid == "true"
        drop = (train = data_drop[idxs1[idx_drop[1]:idx_drop[2]], :], valid = data_drop[idxs1[idx_drop[2]:idx_drop[3]], :], test = data_drop[idxs1[idx_drop[3]:idx_drop[4]], :]);
        drop_std = (train = data_drop_std[idxs1[idx_drop[1]:idx_drop[2]], :], valid = data_drop_std[idxs1[idx_drop[2]:idx_drop[3]], :], test = data_drop_std[idxs1[idx_drop[3]:idx_drop[4]], :]);
        med = (train = data_med[idxs2[idx_med[1]:idx_med[2]], :], valid = data_med[idxs2[idx_med[2]:idx_med[3]], :], test = data_med[idxs2[idx_med[3]:idx_med[4]], :]);
        med_std = (train = data_med_std[idxs2[idx_med[1]:idx_med[2]], :], valid = data_med_std[idxs2[idx_med[2]:idx_med[3]], :], test = data_med_std[idxs2[idx_med[3]:idx_med[4]], :]);
        drop, drop_std, med, med_std
    elseif valid == "false"
        drop = (train = data_drop[idxs1[idx_drop[1]:idx_drop[3]], :], test = data_drop[idxs1[idx_drop[3]:idx_drop[4]], :]);
        drop_std = (train = data_drop_std[idxs1[idx_drop[1]:idx_drop[3]], :], test = data_drop_std[idxs1[idx_drop[3]:idx_drop[4]], :]);
        med = (train = data_med[idxs2[idx_med[1]:idx_med[3]], :], test = data_med[idxs2[idx_med[3]:idx_med[4]], :]);
        med_std = (train = data_med_std[idxs2[idx_med[1]:idx_med[3]], :], test = data_med_std[idxs2[idx_med[3]:idx_med[4]], :]);
        drop, drop_std, med, med_std
    end
end

function generate_all(data, option = "drop")
    if option == "drop"
        drop = dropmissing(data);
        coerce!(drop, :precipitation_nextday => Multiclass);
        drop_std = drop;

        drop, drop_std
    elseif option == "med"
        med = MLJ.transform(fit!(machine(FillImputer(), select(data[:,:], Not([:precipitation_nextday])))), select(data[:,:], Not([:precipitation_nextday])));
        insertcols!(med,:precipitation_nextday=>data.precipitation_nextday[:]);
        coerce!(med, :precipitation_nextday => Multiclass);
        med_std = med;

        med, med_std
    end
end

function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response)
	)
end