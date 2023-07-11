#using SimulatedNeuralMoments, 
using Flux, Test, LinearAlgebra, Statistics
using BSON:@load
cd(@__DIR__)

function main()
@testset "SV" begin
    include("../examples/SV/SVexample.jl")
    chain, θhat, Σp = SVexample(5000, 100) # fast run
    @test size(θhat,1) == 3
    @test size(Σp) == (3,3)
end

@testset "MN" begin
    include("../examples/MN/MNexample.jl")
    chain, θhat, Σp = MNexample(5000, 100) # fast run
    @test size(θhat,1) == 5
    @test size(Σp) == (5,5)
end
end
main()
