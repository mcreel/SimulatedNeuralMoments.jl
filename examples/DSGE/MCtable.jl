using SimulatedNeuralMoments, Statistics, DelimitedFiles
results = readdlm("mcresults.txt")
sofar = 500
println("__________ replication: ", sofar, "_______________")
clabels = ["99%", "95%", "90%"]
prettyprint(reshape(mean(results[1:sofar,8:end],dims=1),7,3),clabels)
dstats(results[1:sofar,1:7])

