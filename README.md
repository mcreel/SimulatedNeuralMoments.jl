# SimulatedNeuralMoments
A package for estimation and inference based on statistics that are filtered through a trained neural net. The methods lead to reliable inferences, in the sense that confidence intervals or credible intervals based on quantiles of MCMC chains contain true parameters at a proportion close to the nominal level, in addition to point estimators having low bias and RMSE. From a Bayesian perspective, the methods can be interpreted as a properly calibrated ABC estimator. From a classical perspective, the methods satisfy the assumptions of Theorem 3 of V. Chernozhukov, H. Hong / Journal of Econometrics 115 (2003) 293 – 346. In either case, the use of a neural net to reduce the dimension of the statistic to the minimum needed to maintain identification is an important factor in obtaining reliable inferences. The evidence is reported in the working paper !["Inference Using Simulated Neural Moments"](https://www.barcelonagse.eu/research/working-papers/inference-using-simulated-neural-moments).

[![Build Status](https://travis-ci.org/mcreel/SimulatedNeuralMoments.jl.svg?branch=main)](https://travis-ci.org/mcreel/SimulatedNeuralMoments.jl)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/docs/API.md)

By way of documentation, there is an explanation of the API in the docs directory, and here's a run through the MNexample.jl file, in examples/MN, which estimates the parameters of a Gaussian Mixture model. This example can serve as a template of how to use the package. See also the README.md in the examples/SV directory.

To use the MNexample.jl file, start Julia from its directory, and, if you do not have MCMCChains and/or StatsPlots installed, activate the environment with ```] activate .``` 

First, we train a neural net to recognize the parameters, given a vector of statistics:

```
using SimulatedNeuralMoments, MCMCChains, StatsPlots
using BSON:@save
using BSON:@load

# get the things to define the structure for the statistical model we wish to estimate
# For your own models, you will need to supply the functions
# found in MNlib.jl, using the same formats
include("MNlib.jl")
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("Mixture of Normals example model", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# train the net, and save it and the transformation info
nnmodel, nninfo = MakeNeuralMoments(model, Epochs=10)
```

Then we can make one sample draw of the statistics at a given parameter vector, and use the statistics to estimate the parameters, to see how well we do:
```
θ = TrueParameters() # this is defined in MNlib.jl

# illustrate basic NN point estimation
m = NeuralMoments(θ, 1, model, nnmodel, nninfo) # the estimate
cnames = ["true", "estimate"]
println("Basic NN estimation, true parameters (a draw from prior) and estimates")
prettyprint([θ m], cnames)

```
The point estimates are:
```
Basic NN estimation, true parameters and estimates
┌──────────────┬──────────────┐
│         true │     estimate │
├──────────────┼──────────────┤
│      1.00000 │      0.73289 │
│      1.00000 │      0.88155 │
│      0.20000 │      0.25068 │
│      1.80000 │      1.72031 │
│      0.40000 │      0.44932 │
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

          μ₁    0.8922    0.0587     0.0006    0.0037   156.7766    1.0049
          μ₂    0.9412    0.1248     0.0012    0.0046   696.6367    1.0002
          σ₁    0.2350    0.0378     0.0004    0.0023   188.1651    1.0056
          σ₂    1.7322    0.0967     0.0010    0.0049   291.5987    1.0019
           p    0.4162    0.0428     0.0004    0.0023   248.6760    1.0039

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

          μ₁    0.7733    0.8535    0.8939    0.9322    1.0001
          μ₂    0.7042    0.8562    0.9406    1.0250    1.1972
          σ₁    0.1600    0.2121    0.2338    0.2586    0.3137
          σ₂    1.5542    1.6682    1.7288    1.7920    1.9381
           p    0.3343    0.3877    0.4154    0.4423    0.5037 
```
The extremum estimator results are:
```
┌──────────────┬──────────────┐
│         true │     estimate │
├──────────────┼──────────────┤
│      1.00000 │      0.96377 │
│      1.00000 │      0.92010 │
│      0.20000 │      0.17996 │
│      1.80000 │      1.77447 │
│      0.40000 │      0.40204 │
└──────────────┴──────────────┘
```
A plot of the chain, and nonparametric plots of marginal posteriors are
![MNchain](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/MN/chain.png)



