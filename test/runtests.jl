#using SimulatedNeuralMoments, 
using Flux, Test, LinearAlgebra
using BSON:@load

@testset "SimulatedNeuralMoments.jl" begin
    include("../examples/SV/SVexample.jl")
end
