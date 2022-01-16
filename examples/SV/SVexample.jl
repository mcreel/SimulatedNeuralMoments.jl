using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles
using BSON:@save
using BSON:@load

# get the things to define the structure for the model
include("SVlib.jl")
function main()
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("Stochastic Volatility example", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# train the net, and save it and the transformation info
#nnmodel, nninfo = MakeNeuralMoments(model)
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# draw a sample at the design parameters, or use an existing data set
#y = SVmodel(TrueParameters()) # draw a sample of 500 obsns. at design parameters
y = readdlm("svdata.txt") # load a data set
p1 = plot(y)
p2 = density(y)
plot(p1, p2, layout=(2,1))
#savefig("data.png")

# define the neural moments using the real data
m = NeuralMoments(auxstat(y), model, nnmodel, nninfo)
@show m
## draw a chain of length 10000 plus 500 burnin
chain, junk, junk = MCMC(m, 10500, model, nnmodel, nninfo, do_cue=true, verbosity=true)
chain = chain[501:end,:]
# visualize results
chn = Chains(chain, ["ϕ", "ρ", "σ"])
display(chn)
println("SNM estimation, estimated pos. median")
cnames = ["pos. median"] 
prettyprint(median(chain,dims=1)[:], cnames)
plot(chn)
savefig("chain.png")
writedlm("chain.txt", chain)
end
main()
