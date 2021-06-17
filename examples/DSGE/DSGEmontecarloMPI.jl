# I recommend starting julia with "julia --project -t X" where X is 
# the number of physical cores available, then, include this file.
using SimulatedNeuralMoments, Flux, SolveDSGE, MPI
using BSON:@save
using BSON:@load

include("CKlib.jl") # contains the functions for the DSGE model
include("Analyze.jl")
include("montecarlo.jl")

function Wrapper()
    lb, ub = PriorSupport()
    model = SNMmodel("DSGE example", lb, ub, InSupport, Prior, PriorDraw, auxstat)
    @load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net
    data = dgp(TrueParameters(), dsge, 1)[1]
    m = NeuralMoments(auxstat(data), model, nnmodel, nninfo)
    chain, junk, junk = MCMC(m, 5500, model, nnmodel, nninfo; verbosity=true)
    vcat(median(chain[501:end,:], dims=1)[:], Analyze(chain))
end
# the monitoring function
function Monitor(sofar, results)
    if mod(sofar,1) == 0
        theta = TrueParameters()
        nParams = size(theta,1)
        println("__________ replication: ", sofar, "_______________")
        println("Results so far")
        println("parameter estimates")
        dstats(results[1:sofar,1:nParams]; short=true)
        println("CI coverage")
        clabels = ["99%","95%","90%"]
        prettyprint(reshape(mean(results[1:sofar,nParams+1:end],dims=1),nParams,3),clabels)
    end
end

function main()
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    reps = 1000   # desired number of MC reps
    n_returns = 28  
    pooled = 1  # do this many reps b
    montecarlo(Wrapper, Monitor, comm, reps, n_returns, pooled)
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

