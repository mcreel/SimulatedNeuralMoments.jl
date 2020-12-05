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
modelname is a string variable, which describes the model.
lb and ub are vectors of Float64. These are needed for the simulated annealing
minimization to obtain the extremum estimator of the model's parameters.
insupport, prior and priordraw and auxstat are all strings, names of functions.

The functions insupport, prior and priordraw are called as, e.g., prior(theta). 

insupport returns a boolean, indicating if the theta is in the InSupport
prior returns a Float64, the prior density, evaluated at theta
priordraw returns a vector of Float64, a draw from the prior at the parameters theta

auxstat is called as auxstat(theta, reps), where reps is Int64. It returns reps draws
of the statistics drawn at theta, in a reps X p array of Float64.

The files MNexample.jl and MNlib.jl show how to use this type.

## The function MakeNeuralMoments

