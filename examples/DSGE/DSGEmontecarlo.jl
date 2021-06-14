# I recommend starting julia with "julia --project -t X" where X is 
# the number of physical cores available, then, include this file.
using SimulatedNeuralMoments, Flux, SolveDSGE
using BSON:@save
using BSON:@load

include("CKlib.jl") # contains the functions for the DSGE model
include("Analyze.jl")
lb, ub = PriorSupport()
model = SNMmodel("DSGE example", lb, ub, InSupport, Prior, PriorDraw, auxstat)

function main()
    lb, ub = PriorSupport()
    model = SNMmodel("Mixture of Normals example model", lb, ub, InSupport, Prior, PriorDraw, auxstat)
    @load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net
    R = 200
    results = zeros(R, 28)
    Threads.@threads for r = 1:R
        data = dgp(TrueParameters())[1]
        m = NeuralMoments(auxstat(data), model, nnmodel, nninfo)
        chain, junk, junk = MCMC(m, 10500, model, nnmodel, nninfo; verbosity=false)
        @show println("r: ", r)
        @show results[r,:] = vcat(median(chain[501:end,:], dims=1)[:], Analyze(chain))
    end
    return results
end

@time results = main()

theta = TrueParameters()
nParams = size(theta,1)
println("Results")
println("parameter estimates (pos. median)")
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

