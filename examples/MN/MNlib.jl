using Statistics, StatsBase

function MNmodel(θ, rndseed=1234)
    n = 1000
    μ1, μ2, σ1, σ2, prob = θ
    d1=randn(n).*σ1 .+ μ1
    d2=randn(n).*(σ1+σ2) .+ (μ1 - μ2) # second component lower mean and higher variance
    ps=rand(n).<prob
    data=zeros(n)
    data[ps].=d1[ps]
    data[.!ps].=d2[.!ps]
    return data
end

function auxstat(data)
    r = 0.1 : 0.1 : 0.9
    vcat(mean(data), std(data), skewness(data), kurtosis(data), 
        quantile.(Ref(data),r))
end

function auxstat(θ, reps)
    auxstat.([MNmodel(θ, rand(1:Int64(1e12))) for i = 1:reps]) 
end

function TrueParameters()
    [1.0, 1.0, 0.2, 1.8, 0.4] # first component N(1,0.2) second component N(0,2)
end    

function PriorSupport()
    lb = [0.0, 0.0, 0.0, 0.0, 0.05] # there is always at least 5% prob for each component  
    ub = [3.0, 3.0, 1.0, 3.0, 0.95] 
    lb,ub
end    

function PriorDraw()
    lb, ub = PriorSupport()
    θ = (ub-lb).*rand(size(lb,1)) + lb
end    

function InSupport(θ)
    lb,ub = PriorSupport()
    all(θ .>= lb) & all(θ .<= ub)
end

# prior is flat over support, so return 1 if inside
function Prior(θ)
    InSupport(θ) ? 1.0 : 0.0
end


