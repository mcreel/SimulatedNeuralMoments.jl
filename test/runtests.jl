#using SimulatedNeuralMoments, 
using Flux, Test, LinearAlgebra
using BSON:@load

@testset "SimulatedNeuralMoments.jl" begin
    include("../examples/SV/SVexample.jl")
    chain, mbar, Σp = main()
    @test size(mbar,1) == 3
    @test size(Σp) == (3,3)
end
