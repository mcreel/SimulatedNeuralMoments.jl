using SimulatedNeuralMoments
using Flux, MCMCChains
using StatsPlots, Distributions
using DelimitedFiles, LinearAlgebra
using BSON:@save
using BSON:@load

# the model-specific code
include("SVlib.jl")

function SVexample(TrainTestSize=1, Epochs=200)

# generate some data, and get sample size 
y = SVmodel(TrueParameters()) # draw a sample at design parameters
n = size(y,1)

# fill in the structure that defines the model
lb, ub = PriorSupport() # bounds of support
model = SNMmodel("Stochastic Volatility example", n, lb, ub, InSupport, Prior, PriorDraw, auxstat)

# train the net, and save it and the transformation info
nnmodel, nninfo = MakeNeuralMoments(model, TrainTestSize=TrainTestSize, Epochs=Epochs)
#  @save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
#  @load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# define the neural moments using the data
θnn = NeuralMoments(auxstat(y), model, nnmodel, nninfo)[:]

# settings
names = ["α", "ρ", "σ", "accept", "lnℒ "]
S = 100
covreps = 1000
length = 5000
burnin = 1000
verbosity = 10 # show results every X draws
tuning = 1.0

# define the proposal
junk, Σp = mΣ(θnn, covreps, model, nnmodel, nninfo)
proposal(θ) = rand(MvNormal(θ, tuning*Σp))

# define the logL
lnL = θ -> snmobj(θ, θnn, S, model, nnmodel, nninfo)
chain = mcmc(θnn, length, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)

Σp = cov(chain[:,1:3])
proposal(θ) = rand(MvNormal(θ, tuning*Σp))
chain = mcmc(θnn, length, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)

# transform back to original domain
chain = Array(chain)
acceptance = mean(chain[:,end-1])
println("acceptance rate: $acceptance")
chain = Chains(chain, names)
display(chain)
display(plot(chain))
savefig("chain.png")
return chain, θnn, Σp
end

