# SimulatedNeuralMoments
[![testing](https://github.com/mcreel/SimulatedNeuralMoments.jl/actions/workflows/testing.yml/badge.svg)](https://github.com/mcreel/SimulatedNeuralMoments.jl/actions/workflows/testing.yml)
(testing is done with the latest stable release of Julia, on Linux, Windows, and MacOS)

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/docs/API.md)

A package for estimation and inference based on statistics that are filtered through a trained neural net. The methods lead to reliable inferences, in the sense that confidence intervals or credible intervals based on quantiles of MCMC chains contain true parameters at a proportion close to the nominal level, in addition to point estimators having low bias and RMSE. The evidence is reported in the paper "Inference Using Simulated Neural Moments" Econometrics 2021, 9(4), 35; https://doi.org/10.3390/econometrics9040035.

The package bases estimation and inference on summary statistics from a simulable model, as is the case with certain versions of Approximate Bayesian Computation (ABC) and the Method of Simulated Moments (MSM). The innovation is a neural net is used to reduce the dimension of the statistics to be the same as that of the parameters of the model (resulting in what are known as just-identifying statistics). The net is trained using many simulated draws from the prior, to get parameters, and the statistics resulting from a sample from the model at each parameter draw. With a large set of draws of parameter/statistics, the parameters are first transformed to lie in ℛⁿ, using Bijectors.jl. The statistics are also regularized. Then, the net is trained, using Flux.jl, using the statistics as the inputs, and the parameters as the outputs. 

With a trained net, we can feed in statistics computed from real data, and obtain estimated parameters that generated the statistics. These estimated parameters then used as the statistics upon which ABC/MSM is based.

ABC/MSM draws from the posterior are done using Turing.jl, concretely MCMC using Metropolis-Hastings sampling. The likelihood is an estimate of the asymptotic Gaussian distribution of the transformed statistics, which is justified due to the central limit theorem applying to the statistics. The proposal density is also the same Gaussian density, estimated at the transformed statistics corresponding to the real data. Because the proposal and the likelihood are identical, at the true parameter values, random walk MH sampling is effective, because MCMC concentrates around the true parameter values, as the sample size becomes large.

Please see the docs, in the link at the top, and the READMEs for the two examples to see how this all works. To run the two examples, using small training and testing samples, do ``using Pkg; Pkg.test("SimulatedNeuralMoments")``
* [Stochastic volatility model](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/SV/README.md)
* [Mixture of normals model](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/MN/README.md)

