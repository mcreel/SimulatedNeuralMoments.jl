# does MCMC using the NN estimate as the statistic
using LinearAlgebra, Statistics #, Optim


# the main MCMC routine, does several short chains to tune proposal
# then a longer final chain
# covreps: replications used to compute weight matrix (the R in the paper, eqn. 5)
function MCMC(θnn, length, model::SNMmodel, nnmodel, nninfo; covreps = 1000, verbosity = false, rt=0.25)
    nParams = size(model.lb,1)
#=    reps = 50 # replications at each trial parameter (the S in the paper, eqn. 6)
    # use a rapid SAMIN to get good initialization values for chain
    obj = θ -> -1.0*H(θ, θnn, reps, model, nnmodel, nninfo) # define the SAMIN criterion
    if verbosity == true
        sa_verbosity = 2
    else
        sa_verbosity = 0
    end
    θsa = (Optim.optimize(obj, model.lb, model.ub, θnn, SAMIN(rt=rt, verbosity=sa_verbosity),Optim.Options(iterations=10^6))).minimizer
=#
    # get covariance estimate using the consistent estimator
    Σ = EstimateΣ(θnn, covreps, model, nnmodel, nninfo) 
    reps = 10 # fewer for the MC chain
    Σinv = inv((1.0+1/reps).*Σ)
    # define things for MCMC
    lnL = θ -> H(θ, θnn, reps, model, nnmodel, nninfo, Σinv)
    ChainLength = 100
    # set up the proposal
    P = ((cholesky(Σ)).U)' # transpose it here 
    # loops to tune proposal
    tuning = 1.0
    MC_loops = 30
    chain = 0.0
    @inbounds for j = 1:MC_loops
        Proposal = θ -> θ + tuning*P*randn(size(θ))
        if j == MC_loops
            ChainLength = length
        end    
        if j > 1 
            θinit = mean(chain[:,1:nParams], dims=1)[:]
        else
            θinit = θnn
        end    
        chain = mcmc(θinit, ChainLength, 0, model.prior, lnL, Proposal, verbosity)
        # adjust tuning to try to keep acceptance rate between 0.25 - 0.35
        if j < MC_loops
            accept = mean(chain[:,end])
            if accept > 0.3
                tuning *= 1.1
            elseif accept < 0.2
                tuning /= 1.1
            end
        end    
    end
    return chain[:,1:nParams], θnn, P, tuning
end
