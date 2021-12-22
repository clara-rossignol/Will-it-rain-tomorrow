using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, MLJ, DataFrames, Random

function generate(data; option = "drop", valid = "true", test = "true")
    new_data = data
    Random.seed!(2809)
    
    if option == "drop"
        new_data = dropmissing!(copy(data));
        coerce!(new_data, :precipitation_nextday => Multiclass);
        idxs = randperm(size(new_data, 1));
        dim = size(new_data)[1];
        if valid == "true"
            if test == "true"
                #If valid is true then a test set is also creater
                idx_valid = Vector{Int}([1, floor(dim/3), floor(2*dim/3), dim]);
                new_data = (train = new_data[idxs[idx_valid[1]:idx_valid[2]], :], valid = new_data[idxs[idx_valid[2]:idx_valid[3]], :], test = new_data[idxs[idx_valid[3]:idx_valid[4]], :]);
                new_data
            else
                "We do not provide a validation set without a test set, look at your arg."
                empty = DataFrame()
                empty
            end
        elseif valid == "false"
            if test == "true"
                idx_test = Vector{Int}([1, floor(dim/2), dim]);
                new_data = (train = new_data[idxs[idx_test[1]:idx_test[2]], :], test = new_data[idxs[idx_test[2]:idx_test[3]], :]);
                new_data
            else
                "Full set transformed"
                new_data
            end
        end
    elseif option == "med"
        new_data = MLJ.transform(fit!(machine(FillImputer(), select(data, Not(:precipitation_nextday)))), select(data, Not(:precipitation_nextday)));
        insertcols!(new_data,:precipitation_nextday=>data.precipitation_nextday[:]);
        coerce!(new_data, :precipitation_nextday => Multiclass);
        idxs = randperm(size(new_data, 1));
        dim = size(new_data)[1];
        if valid == "true"
            if test == "true"
                #If valid is true then a test set is also created
                idx_valid = Vector{Int}([1, floor(dim/3), floor(2*dim/3), dim]);
                new_data = (train = new_data[idxs[idx_valid[1]:idx_valid[2]], :], valid = new_data[idxs[idx_valid[2]:idx_valid[3]], :], test = new_data[idxs[idx_valid[3]:idx_valid[4]], :]);
                new_data
            else
                "We do not provide a validation set without a test set, look at your arg."
                empty = DataFrame()
                empty
            end
        elseif valid == "false"
            if test == "true"
                idx_test = Vector{Int}([1, floor(2*dim/3), dim]);
                new_data = (train = new_data[idxs[idx_test[1]:idx_test[2]], :], test = new_data[idxs[idx_test[2]:idx_test[3]], :]);
                new_data
            elseif test == "false"
                "Full set transformed"
                new_data
            end
        end
    end
end

function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response))
end
