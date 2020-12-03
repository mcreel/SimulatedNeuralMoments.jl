# this will get moved to the module
# define the empty abstract structure for an SNM model

using Pkg
Pkg. activate(".")
using SimulatedNeuralMoments   

# get the things to define the structure for the model
include("examples/MN/MNlib.jl")
lb, ub = PriorSupport()
# fill in the structure
model = SNMmodel("Mixture of Normals example model", lb, ub, InSupport, Prior, PriorDraw, auxstat)
# get the trained net and the transformation info
nnmodel, nninfo = MakeNeuralMoments(model, Epochs=100)
θ = model.priordraw()
m = NeuralMoments(θ, 10, model, nnmodel, nninfo)
