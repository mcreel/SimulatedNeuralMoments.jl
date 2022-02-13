# bounds by quantiles, and standardizes and normalizes around median
using LinearAlgebra
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

# estimate covariance
function EstimateΣ(θ, reps, model::SNMmodel, nnmodel, nninfo)
    z = model.auxstat(θ, reps) 
    cov([sqrt(model.samplesize).*NeuralMoments(z[i], model, nnmodel, nninfo) for i = 1:reps])
end

# moments and covariance
function mΣ(θ, reps, model::SNMmodel, nnmodel, nninfo)
    z = model.auxstat(θ, reps) 
    Zs = sqrt(model.samplesize) .* [NeuralMoments(z[i], model, nnmodel, nninfo) for i = 1:reps]
    m = mean(Zs) 
    c = Symmetric(cov(Zs))
    m, c
end
    

