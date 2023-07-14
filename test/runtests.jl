#using SimulatedNeuralMoments, 
using Test
cd(@__DIR__)

@testset "SimulatedNeuralMoments" begin
    
    @info "running example model with a small sample"
    include("../example/example.jl")
    acceptance, rmse = example(1000, 100, false) # fast run
    @test rmse < 0.2
    @test acceptance > 0.1
    @test acceptance < 0.5
end
