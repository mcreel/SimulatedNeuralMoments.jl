using SimulatedNeuralMoments, Flux, Test
using BSON:@load
cd(@__DIR__)
@testset "SimulatedNeuralMoments.jl" begin
    @load "neuralmodel.bson" nnmodel nninfo
    z = zeros(11)
    @test NeuralMoments(zeros(11), nnmodel, nninfo)[1] ≈ -5.850073134957904 atol = 1e-5
end
