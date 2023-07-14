# API

There are only 4 elements that the package exports: a type that describes the model, and
three functions.

## The SNMmodel type:
```
# the type that holds the model specifics
struct SNMmodel
    modelname::String # name of model
    samplesize::Int64 # number of observations the model generates 
    lb # vector of lower bounds. All parameters must have a finite lower bound.
    ub # vector of upper bounds. All parameters must have a finite upper bound.
    insupport::Function # function that checks if a draw is valid (in bounds)
    prior:: Function # the prior, used to do MCMC
    priordraw::Function # function that returns a draw from the prior
    auxstat::Function # function that returns an array of draws of statistic, or the statistic corresponding to the real data
end
```
One uses this to define the statistical model, the parameters of which we wish to
estimate. This is done as
```
model = SNMmodel(modelname, samplesize, lb, ub, insupport, prior, priordraw, auxstat)
```

### The elements of a SNMmodel type must follow:
* modelname is a string variable, which describes the model.  
* sample size is and integer, the number of data points the model generates
* lb and ub are vectors of Float64. These are the lower and upper bounds of the support of the parameters (the parameter space). At present, only bounded parameter spaces are supported. 
* insupport, prior and priordraw and auxstat are all strings, names of functions.
  * insupport(theta) returns a boolean, indicating if the theta is in the support.
  * prior(theta) computes the prior density at theta
  * priordraw() returns a vector of Float64, a draw from the prior.
* auxstat is called as
    * ```auxstat(theta, reps)```, where reps is Int64. It returns reps draws of the statistics drawn at theta, in a reps dimensional vector of vectors.
    * or as ```auxstat(data)``` where data is an array. This returns the statistics computed at the real data.

The file example.jl the examples directory show how to use this type.

## The function MakeNeuralMoments
```MakeNeuralMoments(model::SNMmodel; TrainTestSize=1, Epochs=1000)```
 The function generates simulated data from the model and trains a neural net to recognize transformed parameters, given the statistic. It returns the trained net, and an array of information for transforming a raw statistic vector from auxstat to use it as an input to the net.
### This function has two required arguments:
* model, which is of type SNMmodel, as discussed above. 
### optional arguments
* TrainTestSize is an optional argument. If omitted, then 20000 samples per parameter
  will be used to train and test the net. The default gives fairly quick training for
  models that are not costly to simulate. For important research results, this number should
  be increased, if computational resources permit.
* Epochs is the number of Flux epochs that will be used to train the net. The default
  gives fairly quick training for models that are not costly to simulate. For important
  results, it is suggested to increase this to 1000 or more, if computational resources
  permit.
### Outputs
* nnmodel: the trained neural net. A Flux.jl chain, which the trained parameters of the net.
* nninfo: the infomation needed to transform a raw vector of statistics, from auxstat, to be used as an input to the neural net.

## The function NeuralMoments
```NeuralMoments(z, model, nnmodel, nninfo```
This function takes a raw statistic, the SNM model structure, the trained neural net, and the array of transforming information, and returns a predicted transformed parameter vector.
### The inputs are
* z: a vector of raw statistics from the model, the output of auxstat(y), for example.
* model: the SNMmodel structure that describes the statistical model being estimated
* nnmodel: the trained neural net, a Flux.jl chain. It is the first of the outputs of MakeNeuralMoments.
* nninfo: a tuple of vectors, used to transform raw statistics, z, to then pass as inputs to nnmodel. nninfo is the second output of MakeNeuralMoments.
### Output
a vector which is the predicted value, from the neural net,  of the transformed parameters that generated the argument z. 

## The function mΣ
```mΣ(θ, reps, model::SNMmodel, nnmodel, nninfo)```
This function computes the mean and covariance matrix of ```reps``` evaluations of  ```NeuralMoments```, where the inputs, ```z```, to ```NeuralMoments``` are drawn at the trial parameter value ```θ```.
### The inputs are
* ```θ```: a trial parameter value
* reps: Int64. The number of simulation draws to use to compute the mean vector and the covariance matrix
* model: an SNMmodel structure that describes the statistical model being estimated
* nnmodel: the trained neural net, a Flux.jl chain. It is the first of the outputs of MakeNeuralMoments.
* nninfo: a tuple of vectors, used to transform raw statistics, z, to then pass as inputs to nnmodel. nninfo is the second output of MakeNeuralMoments.
### Outputs
* mbar: the mean vector of the reps draws from the NeuralMoments function
* Σ: the covariance matrix of the reps draws from the NeuralMoments function

## the function snmobj
```snmobj(θ, m, reps, model::SNMmodel, nnmodel, nninfo)```
This function computes the MSM-CUE log likelihood at the parameter value θ.
### The inputs are
* ```θ```: a trial parameter value
* m: the output of NeuralMoments, when the input z is computed using the real (not simulated) data
* reps: an integer number of replications of the simulated statistics. 
* model: an SNMmodel structure that describes the statistical model being estimated
* nnmodel: the trained neural net, a Flux.jl chain. It is the first of the outputs of MakeNeuralMoments.
* nninfo: a tuple of vectors, used to transform raw statistics, z, to then pass as inputs to nnmodel. nninfo is the second output of MakeNeuralMoments.
### Outputs
the log likelihood, a Float64

## the function mcmc
```mcmc(
    θ, # the initial value
    length,
    lnL, 
    model::SNMmodel,
    nnmodel,
    nninfo,
    proposal,
    burnin::Int=100,
    verbosity::Int=10
    )```
### the inputs are
* θ: the parameter vector used to initialize the chain
* length: the desired length of chain, not counting burnin
* lnL: the MSM-CUE log likelihood, as a closure (see the example)
* model: the SNMmodel type that describes the statistical model being estimated
* nnmodel: the trained neural net, a Flux.jl chain. It is the first of the outputs of MakeNeuralMoments.
* nninfo: a tuple of vectors, used to transform raw statistics, z, to then pass as inputs to nnmodel. nninfo is the second output of MakeNeuralMoments.
* proposal: a function that returns a proposed parameter vector
* burnin: the number of burnin draws to be discarded before saving draws to the chain
* verbosity: an integer: report intermediate results every so often
### Outputs
* chain: a reps X k+2 array. The first k columns hold the k parameters for each draw, the penultimate column holds the log-likelihood value, and the last column holds an indicator as to whether the value is a new accepted value (1) or a copy of the previous value (0). This is simple to facilitate computing the acceptance rate
