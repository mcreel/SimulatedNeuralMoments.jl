using SimulatedNeuralMoments
using Flux, MCMCChains
using StatsPlots, Distributions
using DelimitedFiles, LinearAlgebra
using BSON:@save
using BSON:@load

include("MNlib.jl")
include("runme.jl")

