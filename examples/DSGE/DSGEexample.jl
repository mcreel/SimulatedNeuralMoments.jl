using SimulatedNeuralMoments, Flux, SolveDSGE, MCMCChains, StatsPlots, DelimitedFiles
using BSON:@save
using BSON:@load

# get the things to define the structure for the model
# For your own models, you will need to supply the functions
# found in MNlib.jl, using the same formats
include("CKlib.jl")
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("DSGE example", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# train the net, and save it and the transformation info
#nnmodel, nninfo = MakeNeuralMoments(model)
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# draw a sample at the design parameters, or use the official "real" data
#data = dgp(TrueParameters())
data = readdlm("dsgedata.txt")

# define the neural moments using the real data
z = auxstat(data)
m = min.(max.(Float64.(nnmodel(TransformStats(z', nninfo)')),model.lb),model.ub)
# draw a chain of length 10000 plus 500 burnin
chain, junk, junk = MCMC(m, 10500, model, nnmodel, nninfo, verbosity=false)
chain = chain[501:end,:]
# visualize results
chn = Chains(chain, ["β", "γ", "ρ₁", "σ₁", "ρ₂", "σ₂", "nss"])
display(chn)
plot(chn)
savefig("chain.png")
writedlm("chain.txt", chain)
