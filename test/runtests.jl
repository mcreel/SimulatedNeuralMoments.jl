#using SimulatedNeuralMoments, 
using Test
cd(@__DIR__)

@testset "SimulatedNeuralMoments" begin
    
    @info "running MN example with a small sample"
    include("../examples/MNexample.jl")
    acceptance, rmse = runme(10000,100,false) # fast run
    @test rmse < 0.2
    @test acceptance > 0.1
    @test acceptance < 0.5

    @info "running MN example with a small sample"
    include("../examples/SVexample.jl")
    acceptance, rmse = runme(10000,100,false) # fast run
    @test rmse < 0.2
    @test acceptance > 0.1
    @test acceptance < 0.5

end
