using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles
using Turing, AdvancedMH
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
m = m[:]
@show m

S = 100
covreps = 1000
length = 1000
nchains = 3
burnin = 500
tuning = 10.0

@model function MSM(m, S, model)
    # create the prior: the product of the following array of marginal priors
    θ  ~ arraydist([Uniform(model.lb[i], model.ub[i]) for i = 1:size(model.lb,1)])
    # sample from the model, at the trial parameter value, and compute statistics
    mbar, Σ = SimulatedNeuralMoments.mΣ(θ, S, model, nnmodel, nninfo)
    mbar = vec(mbar)
    m ~ MvNormal(mbar, Σ)
end

# get covariance of proposal
Σ = EstimateΣ(m, covreps, model, nnmodel, nninfo) 
chain = sample(MSM(m, S, model), init_theta=m, 
        MH(:θ => AdvancedMH.RandomWalkProposal(MvNormal(zeros(3), tuning.*Σ))),
        MCMCThreads(), length+burnin, nchains)
 
chain = chain[burnin+1:end,:,:]
end
chain = main()
display(chain)
display(plot(chain))

