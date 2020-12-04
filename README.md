# SimulatedNeuralMoments
package for estimation and inference based on statistics that are filtered through a trained neural net

[![Build Status](https://travis-ci.org/mcreel/SimulatedNeuralMoments.jl.svg?branch=main)](https://travis-ci.org/mcreel/SimulatedNeuralMoments.jl)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/docs/src/Example1.md)

First, we train a neural net to recognize the parameters, given a vector of statistics:

```
using SimulatedNeuralMoments, MCMCChains, StatsPlots
using BSON:@save
using BSON:@load

# get the things to define the structure for the model
# For your own models, you will need to supply the functions
# found in MNlib.jl, using the same formats
include("MNlib.jl")
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("Mixture of Normals example model", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# train the net, and save it and the transformation info
nnmodel, nninfo = MakeNeuralMoments(model, Epochs=10)
```

Then we can make one sample draw of the statistics at a given parameter vector, and use the statistics to estimate the parameters:
```
θ = TrueParameters() # this is defined in MNlib.jl

# illustrate basic NN point estimation
m = NeuralMoments(θ, 10, model, nnmodel, nninfo) # the estimate
cnames = ["true", "estimate"]
println("Basic NN estimation, true parameters (a draw from prior) and estimates")
prettyprint([θ m], cnames)

```
The point estimates are:
```
julia> prettyprint([θ m], cnames)
┌──────────────┬──────────────┐
│         true │     estimate │
├──────────────┼──────────────┤
│      1.00000 │      0.81573 │
│      1.00000 │      0.88172 │
│      0.20000 │      0.20709 │
│      1.80000 │      1.90387 │
│      0.40000 │      0.39846 │
└──────────────┴──────────────┘
```

Then, we sample from the posterior, using the neural net point estimate as the statistic for ABC or GMM-like inference:

```
# draw a chain of length 10000, and get the extremum estimator
chain, θhat = MCMC(m, 10000, model, nnmodel, nninfo, verbosity=true)

# visualize results
chn = Chains(chain, ["μ₁","μ₂","σ₁","σ₂","p"])
display(chn)
plot(chn)
println("SNM estimation, true parameters (a draw from prior) and extremum estimates")
prettyprint([θ θhat], cnames)
```

We obtain

```
julia> display(chn)
Chains MCMC chain (10000×5×1 Array{Float64,3}):

Iterations        = 1:10000
Thinning interval = 1
Chains            = 1
Samples per chain = 10000
parameters        = μ₁, μ₂, σ₁, σ₂, p

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64 

          μ₁    1.0186    0.0642     0.0006    0.0047   117.3583    1.0211
          μ₂    0.9466    0.1403     0.0014    0.0100   138.9072    1.0010
          σ₁    0.1802    0.0407     0.0004    0.0026   169.7718    1.0351
          σ₂    1.8807    0.1054     0.0011    0.0070   157.4444    1.0000
           p    0.4013    0.0444     0.0004    0.0030   161.8545    1.0147

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

          μ₁    0.8897    0.9756    1.0234    1.0653    1.1328
          μ₂    0.6822    0.8491    0.9446    1.0470    1.2299
          σ₁    0.0990    0.1537    0.1808    0.2065    0.2594
          σ₂    1.6866    1.8069    1.8796    1.9519    2.0875
           p    0.3114    0.3707    0.4020    0.4339    0.4829
```

A plot of the chain, and nonparametric plots of marginal posteriors are
![MNchain](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/master/examples/MN/chain.png)



