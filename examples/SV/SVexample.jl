using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles
using Turing, AdvancedMH, LinearAlgebra
using BSON:@save
using BSON:@load

# get the things to define the structure for the model
#include("SNM.jl")
include("SVlib.jl")
include("MakeNeuralMoments.jl")

#function main()
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("Stochastic Volatility example", lb, ub, InSupport, PriorDraw, auxstat)

# train the net, and save it and the transformation info
transf = bijector(@Prior) # transforms draws from prior to draws from  ℛⁿ 
transformed_prior = transformed(@Prior, transf) # the transformed prior
#nnmodel, nninfo = MakeNeuralMoments(model, transf)
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# draw a sample at the design parameters, or use an existing data set
#y = SVmodel(TrueParameters()) # draw a sample of 500 obsns. at design parameters
y = readdlm("svdata.txt") # load a data set
n = size(y,1)
p1 = plot(y)
p2 = density(y)
plot(p1, p2, layout=(2,1))
#savefig("data.png")

# define the neural moments using the real data
m = NeuralMoments(auxstat(y), nnmodel, nninfo)
# the raw NN parameter estimate
θhat = invlink(@Prior, m)
θt_init = m
S = 100
covreps = 1000
length = 2500
nchains = 4
burnin = 500
tuning = 1.5
junk, Σp = mΣ(θhat, covreps, model, nnmodel, nninfo) 

@model function MSM(m, S, model)
    θt ~ transformed_prior
    if !InSupport(invlink(@Prior, θt))
        Turing.@addlogprob! -Inf
        # Exit the model evaluation early
        return
    end    
    # sample from the model, at the trial parameter value, and compute statistics
    mbar, Σ = mΣ(invlink(@Prior,θt), S, model, nnmodel, nninfo)
    m ~ MvNormal(mbar, Symmetric(Σ))
end

chain = sample(MSM(m, S, model),
    MH(:θt => AdvancedMH.RandomWalkProposal(MvNormal(zeros(3), tuning*Σp))),
    MCMCThreads(), length, nchains; init_params=θt_init, discard_initial=burnin)

# single thread
#=
chain = sample(MSM(m, S, model),
    MH(:θt => AdvancedMH.RandomWalkProposal(MvNormal(zeros(3), tuning*Σp))),
    length; init_params = θt_init, discard_initial=burnin)
=#
#end
#chain = main()
display(chain)
display(plot(chain))
chain2 = Array(chain)
acceptance = size(unique(chain2[:,1]),1)[1] / size(chain2,1)
println("acceptance rate: $acceptance")

