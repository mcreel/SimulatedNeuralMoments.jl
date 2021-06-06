# SVexample.jl
The main purpose of this example is to show how to use the methods with real data. To run the file, go to its directory, start Julia, and activate the environment with ```] activate .```  (unless you happen to have all of the dependencies installed in your main environment).

The first block loads packages and sets up the structure that defines the SV model:

```
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
```

Next, we train the net, or use the pre-trained net which I have kindly provided you:
```
# train the net, and save it and the transformation info
#nnmodel, nninfo = MakeNeuralMoments(model)
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net
```
Next, we load some data.  The data was created using the commented lines. Once we have the data, we make some plots, and then we compute the neural
moments:
```
# draw a sample at the design parameters
#θ = TrueParameters()
#y = SVmodel(θ, 500, 100) # draw a sample of 500 obsns. at design parameters (discard 100 burnin observations)
#writedlm("svdata.txt", y)
y = readdlm("svdata.txt") # load a data set
p1 = plot(y)
p2 = density(y)
plot(p1, p2, layout=(2,1))
#savefig("data.png")
```
We define the neural moments:
```
# define the neural moments using the real data
z = auxstat(y)
m = mean(min.(max.(Float64.(nnmodel(TransformStats(z, nninfo)')),model.lb),model.ub),dims=2)
```

The rest of the example is like the mixture of normals example. In the end, we get a MCMC
chain that looks something like
![SVchain](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/SV/chain.png)



