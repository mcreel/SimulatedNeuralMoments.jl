using BSON: @load
using Flux
using StatsPlots
# get the first layer parameters for influence analysis
@load "neuralmodel.bson" nnmodel
beta = nnmodel.layers[1].W # get first layer betas
z = maximum(abs.(beta),dims=1);
heatmap(z, xlabel="statistic", title="Importance of inputs, bright=high, dark=low")
savefig("ImportanceOfStatistics.svg")


