#using SimulatedNeuralMoments, 
using Flux, Test, LinearAlgebra, Statistics
using BSON:@load

@testset "SimulatedNeuralMoments.jl" begin
    include("../examples/SV/SVexample.jl")
    chain, θhat, Σp = SVexample(TrainTestSize=5000, Epochs=200)
    @test size(θhat,1) == 3
    @test size(Σp) == (3,3)
    @test isposdef(cov(chain))
end
