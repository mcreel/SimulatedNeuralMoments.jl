# simulates data from prior, trains and tests the net, and returns
# the trained net and the information for transforming the inputs

using Statistics, Flux
using Base.Iterators

function MakeNeuralMoments(model::SNMmodel;TrainTestSize=1, Epochs=1000)
    data = 0.0
    datadesign = 0.0
    nParams = size(model.lb,1)
    # training and testing
    if (TrainTestSize == 1) TrainTestSize = Int64(2*nParams*1e4); end # use a default size if none provided
    params = zeros(TrainTestSize,nParams)
    statistics = zeros(TrainTestSize,size(model.auxstat(model.lb,1)[1],1))
    Threads.@threads for s = 1:TrainTestSize
        ok = false
        θ = model.priordraw()
        W = (model.auxstat(θ,1))[1]
        # repeat draw if necessary
        while any(isnan.(W)) || !model.gooddata(W)
            θ = model.priordraw()
            W = model.auxstat(θ,1)[1]
        end    
        params[s,:] = θ
        statistics[s,:] = W
    end
    # transform stats to robustify against outliers
    q50 = zeros(size(statistics,2))
    q01 = similar(q50)
    q99 = similar(q50)
    iqr = similar(q50)
    for i = 1:size(statistics,2)
        q = quantile(statistics[:,i],[0.005, 0.25, 0.5, 0.75, 0.995])
        q005[i] = q[1]
        q50[i] = q[3]
        q995[i] = q[5]
        iqr[i] = q[4] - q[2]
    end
    nninfo = (q005, q50, q995, iqr) 
    transf_stats = TransformStats(statistics, nninfo)
    # train net
    TrainingProportion = 0.5 # size of training/testing
    params = Float32.(params)
    s = Float32.(std(params, dims=1)')
    transf_stats = Float32.(transf_stats)
    trainsize = Int(TrainingProportion*TrainTestSize)
    yin = params[1:trainsize, :]'
    yout = params[trainsize+1:end, :]'
    xin = transf_stats[1:trainsize, :]'
    xout = transf_stats[trainsize+1:end, :]'
    # define the neural net
    nStats = size(xin,1)
    NNmodel = Chain(
        Dense(nStats, 10*nParams, hardtanh),
        Dense(10*nParams, 3*nParams, hardtanh),
        Dense(3*nParams, nParams)
    )
    
    # make the batches
    batches = [(xin[:,ind],yin[:,ind])  for ind in partition(1:size(yin,2), 100)]
    
    # define at this scope
    bestmodel = NNmodel # holds best model using validation data
    bestsofar = 1e10 # loss of best model
    opt_state = Flux.setup(Flux.Momentum(), NNmodel)
    oldbest = 1e10
    stagnation = 0
    @info "starting training of the net"
    for epoch = 1:Epochs
        # change from Momentum to AdamW after 20 epochs
        epoch > 20 ? opt_state = Flux.setup(Flux.AdamW(), NNmodel) : nothing
        # do the training
        Flux.train!(NNmodel, batches, opt_state) do m, x, y
            Flux.huber_loss(m(x)./s, y./s, delta=0.1)
        end
        # get validation loss, and update best model if improvement
        current = Flux.huber_loss(NNmodel(xout)./s, yout./s, delta=0.1)
        # keep track of best model
        if current < bestsofar
            bestsofar = current
            bestmodel = NNmodel
        end
        if mod(epoch, 10) == 0
            ingreen = false
            if bestsofar < oldbest
                ingreen = true
                oldbest = bestsofar
                stagnation = 0
            else stagnation += 1    
            end
            stagnation > 100 ? break : nothing
            err = yout - bestmodel(xout)
            rmse = round.((sqrt.(mean(err.^2.0, dims=2)) ./s)', digits=3)
            printstyled("epoch: $epoch, current loss: ", color=:yellow)
            printstyled("$current ", color=:blue)
            printstyled("best loss: ", color=:yellow)
            if ingreen
                printstyled("$bestsofar\n", color=:green)
            else    
                printstyled("$bestsofar\n", color=:blue)
            end
            println("relative rmse.: $rmse")
        end    
    end
    bestmodel, nninfo, params, statistics, transf_stats
end
