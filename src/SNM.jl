# bounds by quantiles, and standardizes and normalizes around median
using LinearAlgebra, Distributions
function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data = max.(data, q01')
    data = min.(data, q99')
    data = (data .- q50') ./ iqr'
    return data
end

# neural moments given statistic
function NeuralMoments(z, model::SNMmodel, nnmodel, nninfo)
    min.(max.(nnmodel(TransformStats((z[:])', nninfo)'), model.lb), model.ub)
end        

# moments and covariance
function mΣ(θ, reps, model::SNMmodel, nnmodel, nninfo)
    z = model.auxstat(θ, reps) 
    Zs = [NeuralMoments(z[i], model, nnmodel, nninfo) for i = 1:reps]
    m = mean(Zs)[:] 
    c = Symmetric(cov(Zs))
    m, c
end

# moments and covariance
function TmΣ(θ, reps, model::SNMmodel, nnmodel, nninfo)
    z = model.auxstat(θ, reps) 
    Zs = D2R.([NeuralMoments(z[i], model, nnmodel, nninfo) for i = 1:reps], model)
    m = mean(Zs)[:] 
    c = Symmetric(cov(Zs))
    m, c
end
     
function D2R(z, model)
    z = (z .- model.lb) ./ model.ub
    z = -log.(1.0 ./ z .- 1.0)
end    
