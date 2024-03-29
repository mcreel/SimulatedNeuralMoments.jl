# The library of functions for the simple SV model

using Statistics, Random

whichdgp = "Stochastic volatility model"

# method that generates the sample
function auxstat(θ, reps)
    auxstat.([dgp(θ, rand(1:Int64(1e12))) for i = 1:reps])  # reps draws of data
end

# method for a given sample
@views function auxstat(y)
	s = std(y)
	y = abs.(y)
	m = mean(y)
	s2 = std(y)
	y = y ./ s2
	k = std((y).^2.0)
	c = cor(y[1:end-1],y[2:end])
	# ratios of quantiles of moving averages to detect clustering
	q = try
	    q = quantile((ma(y,3)[3:end]), [0.25, 0.75])
	catch
	    q = [1.0, 1.0]
	end
	c1 = log(q[2]/q[1])
	stats = sqrt(size(y,1)) .* vcat(m, s, s2, k, c, c1, HAR(y))
end

function dgp(θ, rndseed=1234)
    Random.seed!(rndseed)
    n = 500
    burnin = 100
    ϕ, ρ, σ = θ
    hlag = 0.0
    ys = zeros(n)
    if InSupport(θ)
        @inbounds for t = 1:burnin+n
            h = ρ*hlag + σ*randn()
            if t > burnin 
                ys[t-burnin] = ϕ*exp(h/2.0)*randn()
            end    
            hlag = h
        end
    end    
    ys
end

function TrueParameters()
    [exp(-0.736/2.0), 0.9, 0.363]
end

function PriorSupport()
    lb = [0.05, 0.0, 0.05]
    ub = [2.0, 0.999, 1.0]
    lb,ub
end    

# prior is uniform, so just return a 1 if in support
function Prior(θ)
    InSupport(θ) ? 1.0 : 0.0
end

# check if parameter is in support, and that uncond. var. of vol. is reasonable
function InSupport(θ)
    lb,ub = PriorSupport()
    all(θ .>= lb) & all(θ .<= ub) && (θ[3]/sqrt(1.0 - θ[2]^2.0) < 5.0)   
end

# no data check for this model
function GoodData(z)
    true
end

function PriorDraw()
    lb, ub = PriorSupport()
    ok = false
    θ = 0.0
    while !ok
        θ = (ub-lb).*rand(size(lb,1)) + lb
        ok = InSupport(θ)
    end
    return θ
end    

# taken from https://github.com/mcreel/Econometrics  
# returns the variable (or matrix), lagged p times,
# with the first p rows filled with ones (to avoid divide errors)
# remember to drop those rows before doing analysis
@views function lag(x,p)
	n = size(x,1)
        k = size(x,2)
	lagged_x = [ones(p,k); x[1:n-p,:]]
end

# returns the variable (or matrix), lagged from 1 to p times,
# with the first p rows filled with ones (to avoid divide errors)
# remember to drop those rows before doing analysis
@views function  lags(x,p)
	n = size(x,1)
	k = size(x,2)
	lagged_x = zeros(eltype(x),n,p*k)
	for i = 1:p
		lagged_x[:,i*k-k+1:i*k] = lag(x,i)
	end
    return lagged_x
end	
 
# compute moving average using p most recent values, including current value
@views function ma(x, p)
    m = zeros(size(x))
    for i = p:size(x,1)
        m[i] = mean(x[i-p+1:i])
    end
    return m
end

# auxiliary model: HAR-RV
# Corsi, Fulvio. "A simple approximate long-memory model
# of realized volatility." Journal of Financial Econometrics 7,
# no. 2 (2009): 174-196.
@views function HAR(y)
    ylags = lags(y,10)
    X = [ones(size(y,1)) ylags[:,1]  mean(ylags[:,1:4],dims=2) mean(ylags[:,1:10],dims=2)]
    # drop missings
    y = y[11:end]
    X = X[11:end,:]
    βhat = X\y
    σhat = std(y-X*βhat)     
    vcat(βhat,σhat)
end
