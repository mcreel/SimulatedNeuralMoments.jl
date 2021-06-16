using SimulatedNeuralMoments, Flux, SolveDSGE, MCMCChains, StatsPlots, DelimitedFiles
using BSON:@save
using BSON:@load

# get the things to define the structure for the model
# For your own models, you will need to supply the functions
# found in MNlib.jl, using the same formats
# I recommend starting julia with "julia --project -t X" where X is 
# the number of physical cores available, then, include this file.
include("CKlib.jl") # contains the functions for the DSGE model
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("DSGE example", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# Here, you can train the net from scratch, or use a previous run
# train the net, and save it and the transformation info
#nnmodel, nninfo = MakeNeuralMoments(model, TrainTestSize=700000)  # 1e5 per parameter
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# draw a sample at the design parameters, from the prior, or use the official "real" data
#data = CKdgp(TrueParameters(), dsge, 1)[1]
data = readdlm("dsgedata.txt")

# define the neural moments using the data
m = NeuralMoments(auxstat(data), model, nnmodel, nninfo)
# Here, you can create a new chain, or use the results from a previous run
# draw a chain of length 10000 plus 500 burnin
chain, junk, junk = MCMC(m, 10500, model, nnmodel, nninfo, verbosity=false)
chain = chain[501:end,:]
writedlm("chain.txt", chain)
#chain = readdlm("chain.txt")

# visualize results
chn = Chains(chain, ["β", "γ", "ρ₁", "σ₁", "ρ₂", "σ₂", "nss"])
plot(chn)
savefig("chain.png")
display(chn)

