module SimulatedNeuralMoments
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

include("MakeNeuralMoments.jl")
include("FromEconometrics.jl")
include("SNM.jl")
include("MCMC.jl")
export SNMmodel, MakeNeuralMoments
export prettyprint, dstats
export TransformStats, NeuralMoments, EstimateΣ, H
export MCMC
end
