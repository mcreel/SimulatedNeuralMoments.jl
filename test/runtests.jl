#using SimulatedNeuralMoments, 
using Test
cd(@__DIR__)

@testset "SimulatedNeuralMoments" begin
    
    @info "running SV example with a small sample"
    testmode = ("SVlib.jl", 10000, 100, false)
    include("../examples/Example.jl")
    @test rmse < 0.2
    @test acceptance > 0.1
    @test acceptance < 0.5

    @info "running MN example with a small sample"
    testmode = ("MNlib.jl", 10000, 100, false)
    include("../examples/Example.jl")
    @test rmse < 0.2
    @test acceptance > 0.1
    @test acceptance < 0.5

end
