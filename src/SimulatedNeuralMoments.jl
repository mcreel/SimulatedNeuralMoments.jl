module SimulatedNeuralMoments

# the type that holds the model specifics
struct SNMmodel
    modelname::String # name of model
    samplesize::Int64 # sample size of model
    lb # vector of lower bounds. Can be -Inf, if desired
    ub # vector of upper bounds. Can be inf, if desired
    insupport::Function # function that checks if a draw is valid
    prior::Function # the prior, for MCMC
    priordraw::Function # function that returns a draw from prior
    auxstat::Function # function that returns an array of draws of statistic
end

# define functions 

# bounds by quantiles, and standardizes and normalizes around median
function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data = max.(data, q01')
    data = min.(data, q99')
    data = (data .- q50') ./ iqr'
    return data
end

# neural moments given statistic
function NeuralMoments(z, model::SNMmodel, nnmodel, nninfo)
    min.(max.(Float64.(nnmodel(TransformStats((z[:])', nninfo)')), model.lb), model.ub)
end        

# mean and covariance of output of NN, given draw from prior
using LinearAlgebra
function mΣ(θ, reps, model::SNMmodel, nnmodel, nninfo)
    z = model.auxstat(θ, reps) 
    Zs = [NeuralMoments(z[i], model, nnmodel, nninfo) for i = 1:reps]
    mean(Zs), Symmetric(cov(Zs))
end

# MSM-CUE quasi-log-likelihood, following Chernozhukov and Hong, 2003
using LinearAlgebra
function snmobj(θ, m, reps, model::SNMmodel, nnmodel, nninfo)
    model.insupport(θ) || return -Inf
    mbar, Σ = mΣ(θ, reps, model, nnmodel, nninfo)  
    isposdef(Σ) || return -Inf
    x = m - mbar
    LinearAlgebra.inv!(cholesky!(Σ))
    -0.5*dot(x,Σ,x)
end

include("MakeNeuralMoments.jl")
include("MCMC.jl")
export SNMmodel, MakeNeuralMoments, NeuralMoments, mΣ, mcmc, snmobj
end

