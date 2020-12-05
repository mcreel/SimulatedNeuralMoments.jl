# SimulatedNeuralMoments
A package for estimation and inference based on statistics that are filtered through a trained neural net. The methods lead to reliable inferences, in the sense that confidence intervals or credible intervals contain true parameters at a proportion close to the nominal level, in addition to low bias and RMSE. From a Bayesian perspective, the methods can be interpreted as a properly calibrated ABC estimator. From a classical perspective, the methods satisfy the assumptions of Theorem 3 of V. Chernozhukov, H. Hong / Journal of Econometrics 115 (2003) 293 – 346. The evidence is reported in the working paper !["Inference Using Simulated Neural Moments"](https://www.barcelonagse.eu/research/working-papers/inference-using-simulated-neural-moments).

[![Build Status](https://travis-ci.org/mcreel/SimulatedNeuralMoments.jl.svg?branch=main)](https://travis-ci.org/mcreel/SimulatedNeuralMoments.jl)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/docs/src/Example1.md)

Here's a run through the MNexample.jl file, in examples/MN, which estimates the parameters of a Gaussian Mixture model.

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
Basic NN estimation, true parameters (a draw from prior) and estimates
┌──────────────┬──────────────┐
│         true │     estimate │
├──────────────┼──────────────┤
│      1.00000 │      0.92253 │
│      1.00000 │      0.97220 │
│      0.20000 │      0.21032 │
│      1.80000 │      1.81367 │
│      0.40000 │      0.41895 │
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
Chains MCMC chain (10000×5×1 Array{Float64,3}):

Iterations        = 1:10000
Thinning interval = 1
Chains            = 1
Samples per chain = 10000
parameters        = μ₁, μ₂, σ₁, σ₂, p

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64 

          μ₁    1.0013    0.0648     0.0006    0.0028   428.0129    1.0003
          μ₂    0.9914    0.1372     0.0014    0.0066   382.2044    1.0016
          σ₁    0.1920    0.0357     0.0004    0.0018   282.3489    1.0024
          σ₂    1.8005    0.1047     0.0010    0.0049   409.7620    1.0030
           p    0.3911    0.0485     0.0005    0.0024   304.8917    1.0001

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

          μ₁    0.8790    0.9594    1.0009    1.0440    1.1281
          μ₂    0.7338    0.8964    0.9862    1.0811    1.2720
          σ₁    0.1168    0.1694    0.1950    0.2167    0.2563
          σ₂    1.5918    1.7289    1.7991    1.8685    2.0128
           p    0.2974    0.3572    0.3929    0.4240    0.4787

```

A plot of the chain, and nonparametric plots of marginal posteriors are
![MNchain](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/MN/chain.png)



