




# bounds by quantiles, and standardizes and normalizes around median
function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data = max.(data, q01')
    data = min.(data, q99')
    data = (data .- q50') ./ iqr'
    return data
end

# a draw of neural moments
function NeuralMoments(θ, auxstat, reps, NNmodel, info)
    z = 0.0
    ok = false
    while !ok
        z = auxstat(θ, reps)
        ok = any(isnan.(z))==false
        if !ok "NaN in auxstat, retry" end
    end    
    lb, ub = PriorSupport()
    mean(min.(max.(Float64.(NNmodel(TransformStats(z, info)')),lb),ub),dims=2)
end        

# estimate covariance
function EstimateΣ(θ, reps, auxstat, NNmodel, info)
    ms = zeros(reps, size(θ,1))
    Threads.@threads for i = 1:reps
        ms[i,:] = NeuralMoments(θ, auxstat, 1, NNmodel, info)
    end    
    Σ = cov(ms)
end

# method with identity weight
function H(θ, m, reps, auxstat, NNmodel, info)
    k = size(θ,1)
    invΣ = Matrix(1.0I, k, k)
    H(θ, m, reps, auxstat, NNmodel, info, invΣ)
end    

# log likelihood (GMM-form) with fixed weight matrix
function H(θ, m, reps, auxstat, NNmodel, info, invΣ)
    x = m - NeuralMoments(θ, auxstat, reps, NNmodel, info)
    -0.5*dot(x,invΣ*x)
end

