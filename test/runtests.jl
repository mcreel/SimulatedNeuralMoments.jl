using SimulatedNeuralMoments, Flux, Test, LinearAlgebra
using BSON:@load
cd(@__DIR__)
@testset "SimulatedNeuralMoments.jl" begin
    @load "../examples/SV/neuralmodel.bson" nnmodel nninfo
    include("../examples/SV/SVlib.jl")
    z = zeros(11)
   lb, ub = PriorSupport() # bounds of support
    model = SNMmodel("Stochastic Volatility example", lb, ub, InSupport, PriorDraw, auxstat)
    m, Σp = mΣ((lb+ub)./2.0, 100, model, nnmodel, nninfo)
    @test NeuralMoments(zeros(11), nnmodel, nninfo)[1] ≈ -5.85007 atol = 1e-5
    @test isposdef(Σp)
    @test ishermitian(Σp)
    @test size(Σp,1) == 3
    @test size(m,1) == 3
end
