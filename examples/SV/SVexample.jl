using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles
using BSON:@save
using BSON:@load
using DelimitedFiles

# get the things to define the structure for the model
# For your own models, you will need to supply the functions
# found in MNlib.jl, using the same formats
include("SVlib.jl")
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("Stochastic Volatility example", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# train the net, and save it and the transformation info
#nnmodel, nninfo = MakeNeuralMoments(model)
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# draw a sample at the design parameters
#θ = TrueParameters()
#y = SVmodel(θ, 500, 100) # draw a sample of 500 obsns. at design parameters (discard 100 burnin observations)
#writedlm("svdata.txt", y)
y = readdlm("svdata.txt") # load a data set
p1 = plot(y)
p2 = density(y)
plot(p1, p2, layout=(2,1))
#savefig("data.png")

# define the neural moments using the real data
z = auxstat(y)
m = mean(min.(max.(Float64.(nnmodel(TransformStats(z, nninfo)')),model.lb),model.ub),dims=2)
@show m
# draw a chain of length 10000, and get the extremum estimator (SV model seems to need slow cooling)
chain, θhat = MCMC(m, 10000, model, nnmodel, nninfo, verbosity=false, rt = 0.9)

# visualize results
chn = Chains(chain, ["ϕ", "ρ", "σ"])
display(chn)
println("SNM estimation, extremum estimates")
cnames = ["estimate"] 
prettyprint(θhat, cnames)
plot(chn)
#savefig("chain.png")

