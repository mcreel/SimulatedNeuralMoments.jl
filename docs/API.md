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
  will be used to train an test the net. The default gives fairly quick training for
  models that are not costly to simulate.
  * Epochs is the number of Flux epochs that will be used to train the net. The default
  gives fairly quick training for models that are not costly to simulate.

## The function MCMC
```function MCMC(θnn, length, model::SNMmodel, nnmodel, nninfo; verbosity = false, rt=0.5)```
This function returns a MCMC chain and the extremmum estimator. The arguments are:
  * θnn: A vector of Float64. The value of the neural statistic obtained from the real sample data. 
  * length: Int64. The length of the MCMC chain.
  * model: of type SNMmodel, as discussed above.
  * nnmodel and nninfo: the trained net, and the transformation information. These are obtained as the outputs of MakeNeuralMoments.
  * verbosity: boolean. Controls display of intermediate information.
  * rt: the temperature reduction factor for the simulated annealing minimization that is used to compute
  the extremum estimator.
 

