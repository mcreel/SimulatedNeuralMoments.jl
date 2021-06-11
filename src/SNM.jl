# bounds by quantiles, and standardizes and normalizes around median
function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data = max.(data, q01')
    data = min.(data, q99')
    data = (data .- q50') ./ iqr'
    return data
end

# draw neural moments
function NeuralMoments(θ, reps, model::SNMmodel, nnmodel, nninfo)
    z = model.auxstat(θ, reps) 
    mean([NeuralMoments(z[i], model, nnmodel, nninfo) for i = 1:reps])
end        
# neural moments given statistic
function NeuralMoments(z, model::SNMmodel, nnmodel, nninfo)
    z = (z[:])' # ensure it's a row vector
    min.(max.(Float64.(nnmodel(TransformStats(z, nninfo)')),model.lb),model.ub)[1]
end        


# estimate covariance
function EstimateΣ(θ, reps, model::SNMmodel, nnmodel, nninfo)
    z = model.auxstat(θ, reps) 
    cov([NeuralMoments(z[i], model, nnmodel, nninfo) for i = 1:reps])
end

# method with identity weight
function H(θ, m, reps, model::SNMmodel, nnmodel, nninfo)
    k = size(θ,1)
    invΣ = Matrix(1.0I, k, k)
    H(θ, m, reps, model, nnmodel, nninfo, invΣ)
end    

# log likelihood (GMM-form) with fixed weight matrix
function H(θ, m, reps, model::SNMmodel, nnmodel, nninfo, invΣ)
    x = m - NeuralMoments(θ, reps, model, nnmodel, nninfo)
    -0.5*dot(x,invΣ*x)
end

