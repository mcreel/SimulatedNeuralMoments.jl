# I recommend starting julia with "julia --project -t X" where X is 
# the number of physical cores available, then, include this file.
using SimulatedNeuralMoments, Flux, SolveDSGE, MPI, DelimitedFiles
using BSON:@save
using BSON:@load

include("CKlib.jl") # contains the functions for the DSGE model
include("Analyze.jl")
include("montecarlo.jl")

function Wrapper()
    lb, ub = PriorSupport()
    model = SNMmodel("DSGE example", lb, ub, InSupport, Prior, PriorDraw, auxstat)
    @load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net
    data = dgp(TrueParameters(), dsge, 1, rand(1:Int64(1e12)))[1]
    m = NeuralMoments(auxstat(data), model, nnmodel, nninfo)
    @time chain, junk, junk = MCMC(m, 5500, model, nnmodel, nninfo; verbosity=false, do_cue = true)
    Analyze(chain[501:end,:])
end

function Monitor(sofar, results)
    if mod(sofar,1) == 0
        println("__________ replication: ", sofar, "_______________")
        clabels = ["99%", "95%", "90%"]
        prettyprint(reshape(mean(results[1:sofar,8:end],dims=1),7,3),clabels)
        dstats(results[1:sofar,1:7])
        if size(results,1)==sofar
            writedlm("mcresults.txt", results)
        end    
    end
end

function main()
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    reps = 500
    n_returns = 28 
    pooled = 1
    montecarlo(Wrapper, Monitor, comm, reps, n_returns, pooled)
    MPI.Finalize()
end
@time main()


