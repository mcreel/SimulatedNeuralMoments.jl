using Pkg
Pkg. activate(".")
using SimulatedNeuralMoments   

# get the things to define the structure for the model
include("examples/MN/MNlib.jl")
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("Mixture of Normals example model", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# get the trained net and the transformation info
#nnmodel, nninfo = MakeNeuralMoments(model, Epochs=10)

# illustrate basic NN estimation
θ = model.priordraw() # true parameter
m = NeuralMoments(θ, 10, model, nnmodel, nninfo) # the estimate
cnames = ["true", "estimate"]
println("Basic NN estimation, true parameters (a draw from prior) and estimates")
prettyprint([θ m], cnames)
nothing
