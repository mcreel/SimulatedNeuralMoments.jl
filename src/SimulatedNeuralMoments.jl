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
    samplesize::Int64 # number of observations in data set
end

include("MakeNeuralMoments.jl")
include("SNM.jl")
export SNMmodel, MakeNeuralMoments
export TransformStats, NeuralMoments, EstimateΣ, mΣ
end
