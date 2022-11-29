using SimulatedNeuralMoments
using Flux, Turing, MCMCChains, AdvancedMH
using StatsPlots, DelimitedFiles, LinearAlgebra
using BSON:@save
using BSON:@load

# the model-specific code
include("SVlib.jl")

function main()

# fill in the structure that defines the model
lb, ub = PriorSupport() # bounds of support
model = SNMmodel("Stochastic Volatility example", lb, ub, InSupport, PriorDraw, auxstat)

# train the net, and save it and the transformation info
transf = bijector(@Prior) # transforms draws from prior to draws from  ℛⁿ 
transformed_prior = transformed(@Prior, transf) # the transformed prior
nnmodel, nninfo = MakeNeuralMoments(model, transf, TrainTestSize=10000)
@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# draw a sample at the design parameters, or use an existing data set
y = SVmodel(TrueParameters()) # draw a sample of 500 obsns. at design parameters
#y = readdlm("svdata.txt") # load a data set
n = size(y,1)
p1 = plot(y)
p2 = density(y)
plot(p1, p2, layout=(2,1))
#savefig("data.png")

# define the neural moments using the real data
m = NeuralMoments(auxstat(y), nnmodel, nninfo)
# the raw NN parameter estimate
θhat = invlink(@Prior, m)

# setting for sampling
names = [":α", ":ρ", ":σ"]
S = 100
covreps = 1000
length = 1250
nchains = 4
burnin = 0
tuning = 1.8
# the covariance of the proposal (scaled by tuning)
junk, Σp = mΣ(θhat, covreps, model, nnmodel, nninfo)

@model function MSM(m, S, model)
    θt ~ transformed_prior
    if !InSupport(invlink(@Prior, θt))
        Turing.@addlogprob! -Inf
        return
    end
    # sample from the model, at the trial parameter value, and compute statistics
    mbar, Σ = mΣ(invlink(@Prior,θt), S, model, nnmodel, nninfo)
    m ~ MvNormal(mbar, Symmetric(Σ))
end

chain = sample(MSM(m, S, model),
    MH(:θt => AdvancedMH.RandomWalkProposal(MvNormal(zeros(size(m,1)), tuning*Σp))),
    MCMCThreads(), length, nchains; init_params=Iterators.repeated(m), discard_initial=burnin)

# single thread
#=
chain = sample(MSM(m, S, model),
    MH(:θt => AdvancedMH.RandomWalkProposal(MvNormal(zeros(3), tuning*Σp))),
    length*nthreads; init_params = m, discard_initial=burnin)
=#

# transform back to original domain
chain = Array(chain)
acceptance = size(unique(chain[:,1]),1)[1] / size(chain,1)
println("acceptance rate: $acceptance")
for i = 1:size(chain,1)
    chain[i,:] = invlink(@Prior, chain[i,:])
end
chain = Chains(chain, names)
return chain, mbar, Σp
end
chain, mbar, Σp = main()

display(chain)
display(plot(chain))
#savefig("chain.png")
