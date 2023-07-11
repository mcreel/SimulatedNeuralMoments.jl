#using SimulatedNeuralMoments, 
using Flux, Test, LinearAlgebra, Statistics
using BSON:@load
cd(@__DIR__)

@testset "SV" begin
    @info "running SV model with a small sample"
    include("../examples/SV/SVexample.jl")
    chain, θhat, Σp = SVexample(1000, 100) # fast run
    @test size(θhat,1) == 3
    @test size(Σp) == (3,3)
end

@testset "MN" begin
    @info "running the MN model with a small sample" 
    include("../examples/MN/MNexample.jl")
    chain, θhat, Σp = MNexample(1000, 100) # fast run
    @test size(θhat,1) == 5
    @test size(Σp) == (5,5)
end
