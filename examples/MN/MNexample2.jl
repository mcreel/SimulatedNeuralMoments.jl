# this replicates the results in the paper-
# it can use threads, so set JULIA_NUM_THREADS
# appropriately for your system.
using SimulatedNeuralMoments, Flux
using BSON:@save
using BSON:@load
include("MNlib.jl")
include("Analyze.jl")

function main()
    lb, ub = PriorSupport()
    model = SNMmodel("Mixture of Normals example model", lb, ub, InSupport, Prior, PriorDraw, auxstat)

    # train the net, and save it and the transformation info
    nnmodel, nninfo = MakeNeuralMoments(model)
    @save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
    #@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

    R = 1000
    results = zeros(R, 20)
    Threads.@threads for r = 1:R
        m = NeuralMoments(TrueParameters(), 1, model, nnmodel, nninfo) # the estimate
        chain, θhat, junk, junk = MCMC(m, 10000, model, nnmodel, nninfo)
        println("r: ", r)
        @show results[r,:] = vcat(θhat, Analyze(chain))
    end
    return results
end

@time results = main()

theta = TrueParameters()
nParams = size(theta,1)
println("Results")
println("parameter estimates")
est = results[:,1:nParams]
err = est .- TrueParameters()'
b = mean(err, dims=1)
s = std(err, dims=1)
rmse = sqrt.(b.^2 + s.^2)
prettyprint([b' rmse'], ["bias", "rmse"])
dstats(est; short=true)
println("CI coverage")
clabels = ["99%","95%","90%"]
prettyprint(reshape(mean(results[:,nParams+1:end],dims=1),nParams,3),clabels)

