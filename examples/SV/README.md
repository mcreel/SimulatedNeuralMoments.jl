# SVexample.jl
The main purpose of this example is to show how to use the methods with real data. To run the file, go to its directory, and start Julia using julia --proj, and then instantiate the project to get all needed packages.

## The first block loads packages:
```julia
using SimulatedNeuralMoments
using Flux, Turing, MCMCChains, AdvancedMH
using StatsPlots, DelimitedFiles, LinearAlgebra
using BSON:@save
using BSON:@load
```

## Define the structure for the model
For your own models, you will need to supply the functions found in SVlib.jl, using the same formats
```julia
# fill in the structure that defines the model
lb, ub = PriorSupport() # bounds of support
model = SNMmodel("Stochastic Volatility example", lb, ub, InSupport, PriorDraw, auxstat)
```

## Train the net
or use the pre-trained net which I have kindly provided you. Training the net takes about 10 minute, if you would like to try it.
```julia
# train the net, and save it and the transformation info
transf = bijector(@Prior) # transforms draws from prior to draws from  ℛⁿ 
transformed_prior = transformed(@Prior, transf) # the transformed prior
#nnmodel, nninfo = MakeNeuralMoments(model, transf)
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net
```

## Data
Next, we load some data, either from a file, or by creating new simulated data.
```julia
# draw a sample at the design parameters, or use an existing data set
y = SVmodel(TrueParameters()) # draw a sample of 500 obsns. at design parameters
#y = readdlm("svdata.txt") # load a data set
n = size(y,1)
p1 = plot(y)
p2 = density(y)
plot(p1, p2, layout=(2,1))
```

## Get moments from the data
Next, we set up sampling. We first get the estimated transformed parameters, and the estimated parameters in untransformed form:
```julia
# define the neural moments using the real data
m = NeuralMoments(auxstat(y), nnmodel, nninfo)
# the raw NN parameter estimate
θhat = invlink(@Prior, m)
```

## Set up the controls for MH sampling using Turing:
```julia
# setting for sampling
names = [":α", ":ρ", ":σ"]
S = 100
covreps = 1000
length = 1250
nchains = 4
burnin = 0
tuning = 1.8
# the covariance of the proposal (scaled by tuning)
junk, Σp = mΣ(θhat, covreps, model, nnmodel, nninfo)
```

## Define the likelihood for the Bayesian model
which, in combination with the prior, defines the posterior from which Turing will sample:
```julia
@model function MSM(m, S, model)
    θt ~ transformed_prior
    if !InSupport(invlink(@Prior, θt))
        Turing.@addlogprob! -Inf
        return
    end
    # sample from the model, at the trial parameter value, and compute statistics
    mbar, Σ = mΣ(invlink(@Prior,θt), S, model, nnmodel, nninfo)
    m ~ MvNormal(mbar, Symmetric(Σ))
end
```

## Sample from the posterior
using Metropolis-Hasting, and a random walk multivariate normal proposal. This proposal is effective, because it is an estimate of the asymptotic distribution of the estimated neural moments, m, from above:
```julia
chain = sample(MSM(m, S, model),
    MH(:θt => AdvancedMH.RandomWalkProposal(MvNormal(zeros(size(m,1)), tuning*Σp))),
    MCMCThreads(), length, nchains; init_params=m, discard_initial=burnin)
```

## Transform the parameters of the chain back to the original parameter space:
```julia
chain = Array(chain)
acceptance = size(unique(chain[:,1]),1)[1] / size(chain,1)
println("acceptance rate: $acceptance")
for i = 1:size(chain,1)
    chain[i,:] = invlink(@Prior, chain[i,:])
end
chain = Chains(chain, names)
chain
```

## Results
Finally, we will see something like
![SVsummary](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/SV/summary.png)
![SVchain](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/SV/chain.png)



