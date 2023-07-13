# functions from github.com/mcreel/Econometrics, extracted to not depend on the whole thing.

using Statistics, Printf, Random



# Newey-West covariance estimator
@views function NeweyWest(Z,nlags=0)
#=
    Returns the Newey-West estimator of the asymptotic variance matrix
    INPUTS: Z, a nxk matrix with rows the vector zt'
            nlags, the number of lags
    OUTPUTS: omegahat, the Newey-West estimator of the covariance matrix
=#
    n,k = size(Z)
    # de-mean the variables
    Z .-= mean(Z,dims=1)
    omegahat = Z'*Z/n # sample variance
    # automatic lags?
    if nlags == 0
        nlags = max(0, Int(round(n^0.25)))
    end    
    # sample autocovariances
    for i = 1:nlags
       Zlag = Z[1:n-i,:]
       ZZ = Z[i+1:n,:]
       gamma = (ZZ'*Zlag)/n
       weight = 1.0 - (i/(nlags+1.0))
       omegahat += weight*(gamma + gamma')
    end    
    return omegahat
end


# method using threads and symmetric proposal
@views function mcmc(θ, reps::Int64, burnin::Int64, Prior::Function, lnL::Function, Proposal::Function, report::Bool, nthreads::Int64)
    perthread = reps ÷ nthreads
    chain = zeros(reps, size(θ,1)+1)
    Threads.@threads for t = 1:nthreads # collect the results from the threads
        chain[t*perthread-perthread+1:t*perthread,:] = mcmc(θ, perthread, burnin, Prior, lnL, Proposal, report) 
    end    
    return chain
end

# MH method for symmetric proposal (as is the case with SNM methods)
# the main loop
@views function mcmc(θ, reps::Int64, burnin::Int64, Prior::Function, lnL::Function, Proposal::Function, report::Bool=true)
    reportevery = Int((reps+burnin)/10)
    lnLθ = lnL(θ)
    chain = zeros(reps, size(θ,1)+1)
    naccept = zeros(size(θ))
    for rep = 1:reps+burnin
        θᵗ = Proposal(θ) # new trial value
        if report
            changed = Int.(.!(θᵗ .== θ)) # find which changed
        end    
        # MH accept/reject: only evaluate logL if proposal is in support of prior (avoid crashes)
        pt = Prior(θᵗ)
        accept = false
        if pt > 0.0
            lnLθᵗ = lnL(θᵗ)
            accept = rand() < exp(lnLθᵗ-lnLθ) * pt/Prior(θ)
            if accept
                θ = θᵗ
                lnLθ = lnLθᵗ 
            end
        end
        if report
            naccept .+= changed .* Int.(accept)
        end    
        if (mod(rep,reportevery)==0 && report)
            println("current parameters: ", round.(θ,digits=3))
            println("  acceptance rates: ", round.(naccept/reportevery,digits=3))
            naccept -= naccept
        end    
        if rep > burnin
            chain[rep-burnin,:] = vcat(θ, accept)
        end    
    end
    return chain
end


#
