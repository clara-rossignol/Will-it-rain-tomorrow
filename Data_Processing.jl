using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using CSV, MLJ, DataFrames, Random

function generate(; option = "drop", std = "false", valid = "true", test = "true")

    data = CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame);
    test_all = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame)
    
    Random.seed!(2809)
    
    if option == "drop"
        new_data = dropmissing(data);
        coerce!(new_data, :precipitation_nextday => Multiclass);
        idxs = randperm(size(new_data, 1));
        dim = size(new_data)[1];
        idx = Vector{Int}([1, floor(dim/3), floor(2*dim/3), dim]);
        if std == "true"
            std_mach = machine(Standardizer(), select(new_data, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
            fit!(std_mach)
            new_data = MLJ.transform(std_mach, select(new_data, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
            if valid == "true"
                if test == "true"
                    new_data = (train = new_data[idxs[idx[1]:idx[2]], :], valid = new_data[idxs[idx[2]:idx[3]], :], test = new_data[idxs[idx[3]:idx[4]], :]);
                    new_data
                else
                    new_data = DataFrame()
                    new_data
                end
            else
                if test == "true"
                    new_data = (train = new_data[idxs[idx[1]:idx[3]], :], test = new_data[idxs[idx[3]:idx[4]], :]);
                    new_data
                else
                    new_test = MLJ.transform(std_mach, select(test_all, Not([:ZER_sunshine_1, :ALT_sunshine_4])));
                    new_data, new_test
                end
            end
        else
            if valid == "true"
                if test == "true"
                    new_data = (train = new_data[idxs[idx[1]:idx[2]], :], valid = new_data[idxs[idx[2]:idx[3]], :], test = new_data[idxs[idx[3]:idx[4]], :]);
                    new_data
                else
                    new_data = DataFrame()
                    new_data
                end
            else
                if test == "true"
                    new_data = (train = new_data[idxs[idx[1]:idx[3]], :], test = new_data[idxs[idx[3]:idx[4]], :]);
                    new_data
                else
                    new_data, test_all
                end
            end
        end
    elseif option == "med"
        new_data = MLJ.transform(fit!(machine(FillImputer(), select(data, Not(:precipitation_nextday)))), select(data, Not(:precipitation_nextday)));
        insertcols!(new_data,:precipitation_nextday=>data.precipitation_nextday[:]);
        coerce!(new_data, :precipitation_nextday => Multiclass);
        idxs = randperm(size(new_data, 1));
        dim = size(new_data)[1];
        idx = Vector{Int}([1, floor(dim/3), floor(2*dim/3), dim]);
        if std == "true"
            std_mach = machine(Standardizer(), select(new_data, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
            fit!(std_mach)
            new_data = MLJ.transform(std_mach, select(new_data, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
            if valid == "true"
                if test == "true"
                    #If valid is true then a test set is also created
                    new_data = (train = new_data[idxs[idx[1]:idx[2]], :], valid = new_data[idxs[idx[2]:idx[3]], :], test = new_data[idxs[idx[3]:idx[4]], :]);
                    new_data
                else
                    empty = DataFrame()
                    empty
                end
            else valid == "false"
                if test == "true"
                    idx_test = Vector{Int}([1, floor(2*dim/3), dim]);
                    new_data = (train = new_data[idxs[idx[1]:idx[3]], :], test = new_data[idxs[idx[3]:idx[4]], :]);
                    new_data
                else
                    new_test = MLJ.transform(std_mach, select(test_all, Not([:ZER_sunshine_1, :ALT_sunshine_4])))
                    new_data, new_test
                end
            end
        else
            if valid == "true"
                if test == "true"
                    #If valid is true then a test set is also created
                    new_data = (train = new_data[idxs[idx[1]:idx[2]], :], valid = new_data[idxs[idx[2]:idx[3]], :], test = new_data[idxs[idx[3]:idx[4]], :]);
                    new_data
                else
                    empty = DataFrame()
                    empty
                end
            else
                if test == "true"
                    idx_test = Vector{Int}([1, floor(2*dim/3), dim]);
                    new_data = (train = new_data[idxs[idx[1]:idx[3]], :], test = new_data[idxs[idx[3]:idx[4]], :]);
                    new_data
                else
                    new_data, test_all
                end
            end
        end
    end
end

#Commonly used losses for binary classification
function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response))
end
