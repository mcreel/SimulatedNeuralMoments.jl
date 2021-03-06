using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles
using BSON:@save
using BSON:@load

# get the things to define the structure for the model
# For your own models, you will need to supply the functions
# found in MNlib.jl, using the same formats
include("MNlib.jl")

function main()
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("Mixture of Normals example model", lb, ub, InSupport, Prior, PriorDraw, auxstat)
# train the net, and save it and the transformation info
# nnmodel, nninfo = MakeNeuralMoments(model; Epochs=100)
# @save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net
# illustrate basic NN point estimation
θ = TrueParameters()
m = NeuralMoments(θ, 1, model, nnmodel, nninfo)
cnames = ["true", "estimate"]
println("Basic NN estimation, true parameters and estimates")
prettyprint([θ m], cnames)
# draw a chain of length 10000 plus 500 burnin
chain, junk, junk = MCMC(m, 10500, model, nnmodel, nninfo, verbosity=true)
chain = chain[501:end,:]
# visualize results
chn = Chains(chain, ["μ₁","μ₂","σ₁","σ₂","p"])
display(chn)
plot(chn)
savefig("chain.png")
println("SNM estimation: true params and posterior median")
cnames = ["true", "pos. median"] 
prettyprint([θ median(chain,dims=1)[:]], cnames)
end
main()
