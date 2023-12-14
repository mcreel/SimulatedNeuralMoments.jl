module SimulatedNeuralMoments

# the type that holds the model specifics
struct SNMmodel
    modelname::String # name of model
    samplesize::Int64 # sample size of model
    lb # vector of lower bounds. Can be -Inf, if desired
    ub # vector of upper bounds. Can be inf, if desired
    gooddata::Function # a function that checks if data is good
    insupport::Function # function that checks if a draw is valid
    prior::Function # the prior, for MCMC
    priordraw::Function # function that returns a draw from prior
    auxstat::Function # function that returns an array of draws of statistic
end

# transformation to compactify the statistics, before training the net
function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data = max.(data, q01')
    data = min.(data, q99')
    data = (data .- q50') ./ iqr'
    return data
end

# neural moments given statistic
function NeuralMoments(z, model::SNMmodel, nnmodel, nninfo)
    if model.gooddata(z)
        return min.(max.(Float64.(nnmodel(Float32.(TransformStats((z[:])', nninfo))')), model.lb), model.ub)
    else
        return fill(NaN, size(model.lb,1))
    end    
end        

# mean and covariance of output of NN computed using reps draws at θ
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
    !any(isnan.(mbar)) && !any(isnan.(Σ)) || return -Inf # bad data check
    n = model.samplesize
    Σ *= n*(1+1/reps) # scale for better numerical accuracy
    isposdef(Σ) || return -Inf
    x = sqrt(n)*(m - mbar)
    LinearAlgebra.inv!(cholesky!(Σ))
    -0.5*dot(x,Σ,x)
end

include("MakeNeuralMoments.jl")
include("MCMC.jl")
export SNMmodel, MakeNeuralMoments, NeuralMoments, mΣ, mcmc, snmobj
end

