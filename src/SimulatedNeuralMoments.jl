module SimulatedNeuralMoments

# the type that holds the model specifics
struct SNMmodel
    modelname::String # name of model
    lb # vector of lower bounds. Can be -Inf, if desired
    ub # vector of upper bounds. Can be inf, if desired
    insupport::Function # function that checks if a draw is valid
    priordraw::Function # function that returns a draw from prior
    auxstat::Function # function that returns an array of draws of statistic
end

function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data .= max.(data, q01)
    data .= min.(data, q99)
    data .= (data .- q50) ./ iqr
end

# neural moments given statistic
function NeuralMoments(z, nnmodel, nninfo)
    nnmodel(TransformStats(z, nninfo))
end        

# mean and covariance of output of NN, given draw from prior (not transformed)
# remember that the NN outputs estimates of transformed parameters
using LinearAlgebra
function mΣ(θ, reps, model::SNMmodel, nnmodel, nninfo)
    z = model.auxstat(θ, reps) 
    Zs = [NeuralMoments(z[i], nnmodel, nninfo) for i = 1:reps]
    mean(Zs), Symmetric(cov(Zs))
end

# simulates data from prior, trains and tests the net, and returns
# the trained net and the information for transforming the inputs
using Statistics, Flux
using Base.Iterators
@views function MakeNeuralMoments(model::SNMmodel, transf; TrainTestSize=1, Epochs=1000)
    data = 0.0
    datadesign = 0.0
    nParams = size(model.lb,1)
    # training and testing
    if (TrainTestSize == 1) TrainTestSize = Int64(2*nParams*1e4); end # use a default size if none provided
    params = zeros(nParams, TrainTestSize)
    nStats = size(model.auxstat(model.lb,1)[1],1) 
    statistics = zeros(nStats, TrainTestSize)
    Threads.@threads for s = 1:TrainTestSize
        ok = false
        θ = model.priordraw()
        W = model.auxstat(θ,1)[1]
        # repeat draw if necessary
        while any(isnan.(W))
            θ = model.priordraw()
            W = model.auxstat(θ,1)[1]
        end    
        params[:,s] = transf(θ)
        statistics[:,s] = W
    end
    # transform stats to robustify against outliers
    q50 = zeros(nStats)
    q01 = similar(q50)
    q99 = similar(q50)
    iqr = similar(q50)
    for i = 1:nStats
        q = quantile(statistics[i,:],[0.01, 0.25, 0.5, 0.75, 0.99])
        q01[i] = q[1]
        q50[i] = q[3]
        q99[i] = q[5]
        iqr[i] = q[4] - q[2]
    end
    nninfo = (q01, q50, q99, iqr) 
    for i = 1: nStats
        statistics[:,i] .= TransformStats(statistics[:,i], nninfo)
    end    
    # train net
    TrainingProportion = 0.5 # size of training/testing
    params = Float32.(params)
    s = Float32.(std(params, dims=2))
    statistics = Float32.(statistics)
    trainsize = Int(TrainingProportion*TrainTestSize)
    yin = params[:,1:trainsize]
    yout = params[:,trainsize+1:end]
    xin = statistics[:,1:trainsize]
    xout = statistics[:,trainsize+1:end]
    # define the neural net
    NNmodel = Chain(
        Dense(nStats, 10*nParams, tanh),
        Dense(10*nParams, 3*nParams, tanh),
        Dense(3*nParams, nParams)
    )
    loss(x,y) = Flux.huber_loss(NNmodel(x)./s, y./s; delta=0.1) # Define the loss function
    # monitor training
    function monitor(e)
        println("epoch $(lpad(e, 4)): (training) loss = $(round(loss(xin,yin); digits=4)) (testing) loss = $(round(loss(xout,yout); digits=4))| ")
    end
    # do the training
    bestsofar = 1.0e10
    pred = 0.0 # define it here to have it outside the for loop
    batches = [(xin[:,ind],yin[:,ind])  for ind in partition(1:size(yin,2), 50)]
    bestmodel = 0.0
    for i = 1:Epochs
        if i < 20
            opt = Momentum() # the optimizer
        else
            opt = Adam() # the optimizer
        end 
        Flux.train!(loss, Flux.params(NNmodel), batches, opt)
        current = loss(xout,yout)
        if current < bestsofar
            bestsofar = current
            bestmodel = NNmodel
            xx = xout
            yy = yout
            monitor(i)
            pred = NNmodel(xx)
            error = yy .- pred
            results = [pred;error]
            rmse = sqrt.(mean(error.^Float32(2.0),dims=2))
            @info "RMSE for model parameters $rmse" 
        end
    end
    bestmodel, nninfo
end

export SNMmodel, MakeNeuralMoments, NeuralMoments, mΣ
end

