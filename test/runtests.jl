#using SimulatedNeuralMoments, 
using Flux, Test, LinearAlgebra, Statistics
using BSON:@load

@testset "SimulatedNeuralMoments.jl" begin
    include("../examples/SV/SVexample.jl")
    chain, θhat, Σp = SVexample(5000, 200) # fast run
    @test size(θhat,1) == 3
    @test size(Σp) == (3,3)
end
