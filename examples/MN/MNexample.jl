using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots
using BSON:@save
using BSON:@load

# get the things to define the structure for the model
# For your own models, you will need to supply the functions
# found in MNlib.jl, using the same formats
include("MNlib.jl")
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("Mixture of Normals example model", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# train the net, and save it and the transformation info
nnmodel, nninfo = MakeNeuralMoments(model; Epochs=100)
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
#@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# define the true parameter vector (either a specific design or a random value
# θ = model.priordraw() # true parameter is a draw from prior
θ = TrueParameters() # this is defined in MNlib.jl

# illustrate basic NN point estimation
m = NeuralMoments(θ, 1, model, nnmodel, nninfo) # the estimate
cnames = ["true", "estimate"]
println("Basic NN estimation, true parameters and estimates")
prettyprint([θ m], cnames)

# draw a chain of length 10000, and get the extremum estimator
chain, θhat = MCMC(m, 10000, model, nnmodel, nninfo, verbosity=true)

# visualize results
chn = Chains(chain, ["μ₁","μ₂","σ₁","σ₂","p"])
display(chn)
plot(chn)
println("SNM estimation, true parameters and extremum estimates")
prettyprint([θ θhat], cnames)


