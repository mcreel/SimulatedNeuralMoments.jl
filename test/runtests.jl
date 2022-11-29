#using SimulatedNeuralMoments, 
using Flux, Test, LinearAlgebra
using BSON:@load

@testset "SimulatedNeuralMoments.jl" begin
    include("../examples/SV/SVexample.jl")
    chain, θhat, Σp = SVexample(TrainTestSize=5000, Epochs=200)
    @test size(θhat,1) == 3
    @test size(Σp) == (3,3)
    @tesit isposdef(cov(chain))
end
