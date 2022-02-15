module SimulatedNeuralMoments
using LinearAlgebra
include("MakeNeuralMoments.jl")

# the type that holds the model specifics
struct SNMmodel
    modelname::String # name of model
    lb::Vector # vector of lower bounds. Can be -Inf, if desired
    ub::Vector # vector of upper bounds. Can be inf, if desired
    insupport::Function # function that checks if a draw is valid
    prior::Function # function that evaluates the prior at draw
    priordraw::Function # function that returns a draw from prior
    auxstat::Function # function that returns an array of draws of statistic
    samplesize::Int64 # number of observations in data set
end

function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data = max.(data, q01')
    data = min.(data, q99')
    data = (data .- q50') ./ iqr'
end

# neural moments given statistic
function NeuralMoments(z, model::SNMmodel, nnmodel, nninfo)
    min.(max.(nnmodel(TransformStats((z[:])', nninfo)'), model.lb), model.ub)[:]
end        

# moments and covariance
function mΣ(θ, reps, model::SNMmodel, nnmodel, nninfo, transform=true)
    z = model.auxstat(θ, reps) 
    if transform 
        Zs = [D2R(NeuralMoments(z[i], model, nnmodel, nninfo), model) for i = 1:reps]
    else
        Zs = [NeuralMoments(z[i], model, nnmodel, nninfo) for i = 1:reps]
    end
    mean(Zs)[:], Symmetric(cov(Zs))
end

# maps from parameter space to Euclidean space
function D2R(z, model)
    z .-= model.lb
    z ./= model.ub
    z .*=  0.9999
    z .+0 0.00005
    z .= log.(z ./(1.0 .-z))
end

export SNMmodel, MakeNeuralMoments
export TransformStats, NeuralMoments, EstimateΣ, mΣ, D2R
end
