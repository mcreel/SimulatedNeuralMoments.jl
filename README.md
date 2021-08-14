# SimulatedNeuralMoments

[![Build Status](https://travis-ci.org/mcreel/SimulatedNeuralMoments.jl.svg?branch=main)](https://travis-ci.org/mcreel/SimulatedNeuralMoments.jl)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/docs/API.md)

A package for estimation and inference based on statistics that are filtered through a trained neural net. The methods lead to reliable inferences, in the sense that confidence intervals or credible intervals based on quantiles of MCMC chains contain true parameters at a proportion close to the nominal level, in addition to point estimators having low bias and RMSE. From a Bayesian perspective, the methods can be interpreted as a properly calibrated Approximate Bayesian Computing estimator, in the sense of Fearnhead, P. and Prangle, D. (2012), Journal of the Royal Statistical Society: Series B (Statistical Methodology), 74: 419-474. From a frequentist perspective, the methods may be interpreted as a simulated method of moments estimator implemented using the methods of V. Chernozhukov, H. Hong (2003) Journal of Econometrics 115 (2003) 293 – 346. The approximate likelihood that is used for MCMC sampling is the asymptotic Gaussian likelihood of the selected statistics, and this satisfies the generalized information equality requirement of Theorem 3 of Chernozhukov and Hong. Finally, the use of a neural net to reduce the dimension of the vector of statistics to the minimum needed to maintain identification is an important factor in obtaining reliable inferences. The evidence is reported in the working paper "Inference Using Simulated Neural Moments" which is available at https://github.com/mcreel/SNM/blob/master/SNM.pdf.

For the Mixture of Normals model presented below, the confidence interval coverage for
1000 replications is 
![MNmontecarlo](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/MN/montecarlo.png)

which verifies that, for this model and this experimental design, the methods lead to reliable inferences on parameters. These results can be replicated using the MNexample2.jl script in the examples/MN directory. The working paper referenced above contains additional examples which also confirm the reliability of inferences.

By way of documentation, there is an explanation of the API in the docs directory, and here's a run through the MNexample.jl file, in examples/MN, which estimates the parameters of a Gaussian Mixture model. This example can serve as a template of how to use the package. See also the README.md in the examples/SV directory.

To use the MNexample.jl file, start Julia from its directory, and, if you do not have MCMCChains and/or StatsPlots installed, activate the environment with ```] activate .``` 

First, we train a neural net to recognize the parameters, given a vector of statistics:

```julia
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
```julia
θ = TrueParameters() # this is defined in MNlib.jl

# illustrate basic NN point estimation
m = NeuralMoments(θ, 1, model, nnmodel, nninfo) # the estimate
```

Then, we sample from the posterior, using the neural net point estimate as the statistic for ABC or GMM-like inference:
```julia
# draw a chain of length 10000, and get the extremum estimator
chain, θhat = MCMC(m, 10000, model, nnmodel, nninfo, verbosity=true)

# visualize results
chn = Chains(chain, ["μ₁","μ₂","σ₁","σ₂","p"])
display(chn)
plot(chn)
println("SNM estimation, true parameters (a draw from prior) and extremum estimates")
prettyprint([θ θhat], cnames)
```
For reference, the true parameter values are
```julia
julia> TrueParameters()'
1×5 adjoint(::Vector{Float64}) with eltype Float64:
 1.0  1.0  0.2  1.8  0.4
 ```

The estimation results are

![MNresults](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/MN/results.png)

A plot of the chain, and nonparametric plots of marginal posteriors are
![MNchain](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/MN/chain.png)



