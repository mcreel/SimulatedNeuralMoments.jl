# This does extremum GMM and then MCMC using the NN estimate as the statistic
using Flux, Econometrics, LinearAlgebra, Statistics, DelimitedFiles


# the main MCMC routine, does several short chains to tune proposal
# then a longer final chain
function MCMC(θnn, auxstat, NNmodel, info; verbosity = false, nthreads=1, rt=0.5)
    lb, ub = PriorSupport()
    nParams = size(lb,1)
    reps = 10 # replications at each trial parameter (the S in the paper, eqn. 6)
    covreps = 500 # replications used to compute weight matrix (the R in the paper, eqn. 5)
    # use a rapid SAMIN to get good initialization values for chain
    obj = θ -> -1.0*H(θ, θnn, 10, auxstat, NNmodel, info) # define the SAMIN criterion
    if verbosity == true
        sa_verbosity = 2
    else
        sa_verbosity = 0
    end    
    θsa, junk, junk, junk = samin(obj, θnn, lb, ub; coverage_ok=0, maxevals=1000, verbosity = sa_verbosity, rt = rt)
    # get covariance estimate using the consistent estimator
    Σ = EstimateΣ(θsa, covreps, auxstat, NNmodel, info) 
    Σinv = inv((1.0+1/reps).*Σ)
    # define things for MCMC
    lnL = θ -> H(θ, θnn, reps, auxstat, NNmodel, info, Σinv)
    ChainLength = Int(1000/nthreads) # usually, nthreads will be 1, this is only for costly models
    # set up the proposal
    P = 0.0
    try
        P = ((cholesky(Σ)).U)' # transpose it here 
    catch
        P = diagm(sqrt.(diag(Σ)))
    end
    tuning = 1.0
    Proposal = θ -> θ + tuning*P*randn(size(θ))
    # initial short chain to tune proposal
    chain = mcmc(θsa, ChainLength, 0, Prior, lnL, Proposal, verbosity, nthreads)
    # loops to tune proposal
    Σ = NeweyWest(chain[:,1:nParams])
    MC_loops = 5
    @inbounds for j = 1:MC_loops
        P = try
            P = ((cholesky(Σ)).U)'
        catch
            P = diagm(sqrt.(diag(Σ)))
        end    
        Proposal = θ -> θ + tuning*P*randn(size(θ))
        if j == MC_loops
            ChainLength = Int(10000/nthreads)
        end    
        θinit = mean(chain[:,1:nParams],dims=1)[:] # start where last chain left off
        chain = mcmc(θinit, ChainLength, 0, Prior, lnL, Proposal, verbosity, nthreads)
        # adjust tuning to try to keep acceptance rate between 0.23 - 0.35
        if j < MC_loops
            accept = mean(chain[:,end])
            if accept > 0.35
                tuning *= 1.5
            elseif accept < 0.25
                tuning *= 0.25
            end
            Σ = 0.5*Σ + 0.5*NeweyWest(chain[:,1:nParams]) # gradual adjustment to stay on tracks
        end    
    end
    return chain[:,1:nParams], θsa
end
