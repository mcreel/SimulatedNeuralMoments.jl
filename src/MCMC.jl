# simple MH MCMC for symmetric proposal
@views function mcmc(
    θ, # the initial value
    length,
    lnL, 
    model::SNMmodel,
    nnmodel,
    nninfo,
    proposal,
    burnin::Int=100,
    verbosity::Int=10
    )
    Lₙθ = lnL(θ) # Objective at current params
    pθ = model.prior(θ)
    naccept = 0 # Number of acceptance / rejections
    accept = false
    acceptance_rate = 1f0
    chain = zeros(length, size(θ, 1) + 2)
    totreps = length+burnin
    for i ∈ 1:totreps
        θᵗ = proposal(θ) # new trial value
        pθᵗ = model.prior(θᵗ) # prior at trial
        Lₙθᵗ = lnL(θᵗ) # objective trial value
        # Accept / reject trial value
        accept = rand() < exp(Lₙθᵗ - Lₙθ) * pθᵗ/pθ 
        if accept
            # Replace values
            θ = θᵗ
            Lₙθ = Lₙθᵗ
            pθ = pθ 
            # Increment number of accepted values
            naccept += 1
        end
        # Add to chain if burnin is passed
        if i > burnin
            chain[i-burnin,:] = vcat(θ, accept, Lₙθ)
        end
        # Report
        if verbosity > 0 && mod(i, verbosity) == 0 && i > burnin
            acceptance_rate = naccept / i
            println("iter $i of $totreps  params: ", round.(θ, digits=3), " acc.: ", round(acceptance_rate, digits=3))
        end
    end
    return chain
end
