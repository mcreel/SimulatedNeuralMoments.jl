#using SimulatedNeuralMoments, 
using Flux, Test, LinearAlgebra
using BSON:@load

@testset "SimulatedNeuralMoments.jl" begin
    include("../examples/SV/SVexample.jl")
    chain, θhat, Σp = main()
    @test size(θhat,1) == 3
    @test size(Σp) == (3,3)
end
