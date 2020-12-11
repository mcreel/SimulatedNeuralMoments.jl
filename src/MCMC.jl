# This does extremum GMM and then MCMC using the NN estimate as the statistic
using LinearAlgebra, Statistics, Optim


# the main MCMC routine, does several short chains to tune proposal
# then a longer final chain
function MCMC(θnn, length, model::SNMmodel, nnmodel, nninfo; verbosity = false, rt=0.25)
    nParams = size(model.lb,1)
    reps = 10 # replications at each trial parameter (the S in the paper, eqn. 6)
    covreps = 500 # replications used to compute weight matrix (the R in the paper, eqn. 5)
    # use a rapid SAMIN to get good initialization values for chain
    obj = θ -> -1.0*H(θ, θnn, reps, model, nnmodel, nninfo) # define the SAMIN criterion
    if verbosity == true
        sa_verbosity = 2
    else
        sa_verbosity = 0
    end
    θsa = (Optim.optimize(obj, model.lb, model.ub, θnn, SAMIN(rt=rt, verbosity=sa_verbosity),Optim.Options(iterations=10^6))).minimizer
    # get covariance estimate using the consistent estimator
    Σ = EstimateΣ(θsa, covreps, model, nnmodel, nninfo) 
    Σinv = inv((1.0+1/reps).*Σ)
    # define things for MCMC
    lnL = θ -> H(θ, θnn, reps, model, nnmodel, nninfo, Σinv)
    ChainLength = 1000
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
    chain = mcmc(θsa, ChainLength, 0, model.prior, lnL, Proposal, verbosity)
    Σ = NeweyWest(chain[:,1:nParams])
    # loops to tune proposal
    MC_loops = 5
    @inbounds for j = 1:MC_loops
        if j == MC_loops
            ChainLength = length
        end    
        θinit = mean(chain[:,1:nParams],dims=1)[:] # start where last chain left off
        chain = mcmc(θinit, ChainLength, 0, model.prior, lnL, Proposal, verbosity)
        # adjust tuning to try to keep acceptance rate between 0.25 - 0.35
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
