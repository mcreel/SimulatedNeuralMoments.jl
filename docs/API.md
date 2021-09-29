# API

There are only 3 elements that the package provides: a type that describes the model, and
two functions.

## The SNMmodel type:
```
# the type that holds the model specifics
struct SNMmodel
    modelname::String # name of model
    lb::Vector # vector of lower bounds. Can be -Inf, if desired
    ub::Vector # vector of upper bounds. Can be inf, if desired
    insupport::Function # function that checks if a draw is valid
    prior::Function # function that evaluates the prior at draw
    priordraw::Function # function that returns a draw from prior
    auxstat::Function # function that returns an array of draws of statistic
end
```
One uses this to define the statistical model, the parameters of which we wish to
estimate. This is done as
```
model = SNMmodel(modelname, lb, ub, insupport, prior, priordraw, auxstat)
```

The elements of a SNMmodel type must follow:
* modelname is a string variable, which describes the model.
* lb and ub are vectors of Float64. These are needed for the simulated annealing
minimization to obtain the extremum estimator of the model's parameters.
* insupport, prior and priordraw and auxstat are all strings, names of functions. They are called as, e.g., ```prior(theta)```. 
  * insupport returns a boolean, indicating if the theta is in the InSupport
  * prior returns a Float64, the prior density, evaluated at theta
  * priordraw returns a vector of Float64, a draw from the prior at the parameters theta
* auxstat is called as ```auxstat(theta, reps)```, where reps is Int64.
  * It returns reps draws of the statistics drawn at theta, in a reps X p array of Float64.

The files MNexample.jl and MNlib.jl in the examples/MN directory show how to use this type.

## The function MakeNeuralMoments
```function MakeNeuralMoments(model::SNMmodel;TrainTestSize=1, Epochs=1000)```
This function has only one required argument, model, which is of type SNMmodel, as
discussed above. The function generates simulated data from the model and trains a neural
net to recognize the parameters, given the statistic. It returns the trained net, and an
array of information for transforming a raw statistic vector to use it as an input to the
net.
  * TrainTestSize is an optional argument. If omitted, then 20000 samples per parameter
  will be used to train and test the net. The default gives fairly quick training for
  models that are not costly to simulate. For important research results, this number should
  be increased, if computational resources permit.
  * Epochs is the number of Flux epochs that will be used to train the net. The default
  gives fairly quick training for models that are not costly to simulate. For important
  results, it is suggested to increase this to 1000 or more, if computational resources
  permit.

## The function MCMC
```function MCMC(θnn, length, model::SNMmodel, nnmodel, nninfo; covreps = 1000, tuningloops = 10, verbosity = false, do_cue = false, burnin = 0, nthreads=1, tuning = 1.0) ```
This function returns a MCMC chain and the extremmum estimator. The arguments are:
  * θnn: A vector of Float64. The starting value for the chain. It is recommended to use the neural statistic obtained from the real sample data. 
  * length: Int64. The length of the MCMC chain.
  * model: of type SNMmodel, as discussed above.
  * nnmodel and nninfo: the trained net, and the transformation information. These are obtained as the outputs of MakeNeuralMoments.
  * covreps: Int64. The number of replications used to compute the covariance matrix of the proposal density.
  * tuning loops: Int64. The number of short MCMC chains generated to tune the proposal density.
  * verbosity: boolean. Controls display of intermediate information.
  * burnin: Int64. Number of MCMC steps discarded from chain.
  * do_cue: boolean. Whether or not to use continuously updating GMM criterion. Default is no, in which case the two-step version of the GMM
      criterion is used. The two-step version is considerably faster, but the CUE version leads to more accurate confidence intervl coverage, and
      is recommended when it is computationally feasible.
 

