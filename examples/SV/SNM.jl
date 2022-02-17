

# the type that holds the model specifics
struct SNMmodel
    modelname::String # name of model
    lb::Vector # vector of lower bounds. Can be -Inf, if desired
    ub::Vector # vector of upper bounds. Can be inf, if desired
    insupport::Function # function that checks if a draw is valid
    prior::Function # function that evaluates the prior at draw
    priordraw::Function # function that returns a draw from prior
    auxstat::Function # function that returns an array of draws of statistic
    samplesize::Int64 # number of observations in data set
end

#include("MakeNeuralMoments.jl")

function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data .= max.(data, q01)
    data .= min.(data, q99)
    data .= (data .- q50) ./ iqr
end

# neural moments given statistic
function NeuralMoments(z, model::SNMmodel, nnmodel, nninfo)
    min.(max.(nnmodel(TransformStats!((z[:]), nninfo)), model.lb), model.ub)
end        

# moments and covariance
using LinearAlgebra
function mΣ(θ, reps, model::SNMmodel, nnmodel, nninfo, transform=true)
    z = model.auxstat(θ, reps) 
    if transform 
        Zs = [D2R(NeuralMoments(z[i], model, nnmodel, nninfo), model) for i = 1:reps]
    else
        Zs = [NeuralMoments(z[i], model, nnmodel, nninfo) for i = 1:reps]
    end
    mean(Zs), Symmetric(cov(Zs))
end

# maps from parameter space to ℛ 
function D2R(z, model)
    # map to (eps/2,1-eps/2)
    eps = 1e-5
    z .-= model.lb
    z ./= model.ub
    z .*=  (1.0 - eps)
    z .+ eps/2.
    # now, map to ℛ, but with bulletproofing
    z .= log.(z ./(1.0 .-z))
end

# maps from ℛ to original bounded parameter space
function R2D(z, model)
    eps = 1e-5
    z .= exp.(z) ./ (1.0 .+ exp.(z))
    z .-= eps/2.
    z ./= (1.0 - eps)
    z .*= model.ub
    z .+= model.lb
end    


using PrettyTables, Printf

function prettyprint(a, cnames="", rnames="")
if rnames !=""
    rnames = rnames[:]
    a = [rnames a]
    if cnames != ""
        cnames = cnames[:]
        cnames = vcat("", cnames)
    end    
end
if cnames !=""
    pretty_table(a; header=(cnames), formatters=ft_printf("%12.5f"))
else
    pretty_table(a; formatters=ft_printf("%12.5f"))
end
end

function dstats(x, rnames="";short=false, silent=false)
    k = size(x,2)
    if rnames==""
        rnames = 1:k
        rnames = rnames'
    end
    m = mean(x,dims=1)
    mm = median(x,dims=1)
    s = std(x,dims=1)
    sk = m-m
    kt = m-m
    mn = minimum(x,dims=1)
    mx = maximum(x,dims=1)
    q05 = fill(0.0,k)
    q25 = fill(0.0,k)
    q75 = fill(0.0,k)
    q95 = fill(0.0,k)
    if short == false
        for i = 1:size(x,2) q05[i], q25[i], q75[i],q95[i] = quantile(x[:,i], [0.05,0.25,0.75,0.95]) end
        cnames = ["  mean", " median","  std", "IQR", "min", "max", "q05", "q95"]
        stats = [m' mm' s' (q75-q25) mn' mx' q05 q95] 
        if !silent prettyprint(stats, cnames, rnames) end
    else
        cnames = ["  mean", " median", "  std", "min", "max"]
        stats = [m' mm' s' mn' mx'] 
        if !silent prettyprint(stats, cnames, rnames) end
    end
    return stats
end
