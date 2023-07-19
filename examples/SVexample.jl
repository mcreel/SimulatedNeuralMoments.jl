using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.update()
Pkg.instantiate()
using SimulatedNeuralMoments
using Flux, MCMCChains
using StatsPlots, Distributions
using DelimitedFiles, LinearAlgebra
using BSON:@save
using BSON:@load

include("SVlib.jl")
include("runme.jl")

