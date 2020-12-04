using SimulatedNeuralMoments
using BSON:@save
using BSON:@load

# get the things to define the structure for the model
include("MNlib.jl")
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("Mixture of Normals example model", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# get the trained net and the transformation info
nnmodel, nninfo = MakeNeuralMoments(model, Epochs=10)
@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
#@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# illustrate basic NN estimation
θ = model.priordraw() # true parameter
m = NeuralMoments(θ, 10, model, nnmodel, nninfo) # the estimate
cnames = ["true", "estimate"]
println("Basic NN estimation, true parameters (a draw from prior) and estimates")
prettyprint([θ m], cnames)

# draw a chain of length 10000
chain, θhat = MCMC(m, 10000, model, nnmodel, nninfo, verbosity=false)
