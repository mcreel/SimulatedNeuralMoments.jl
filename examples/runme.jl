function runme(TrainTestSize=1, Epochs=1000, saveplot=false)

# generate some data, and get sample size 
y = dgp(TrueParameters()) # draw a sample at design parameters
n = size(y,1)

# fill in the structure that defines the model
lb, ub = PriorSupport() # bounds of support
model = SNMmodel(whichdgp, n, lb, ub, InSupport, Prior, PriorDraw, auxstat)

# train the net, and save it and the transformation info
nnmodel, nninfo, params, stats, transf_stats = MakeNeuralMoments(model, TrainTestSize=TrainTestSize, Epochs=Epochs)

# example transformed stats to ensure that outliers
# have been controlled. We want to see some distance between the whiskers.
for i = 1:size(transf_stats,2)
    display(boxplot(transf_stats[:,i],title="statistic $i"))
    sleep(3)
end

#  @save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
#  @load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# define the neural moments using the data
θnn = NeuralMoments(auxstat(y), model, nnmodel, nninfo)[:]

# settings for MCMC
whichdgp == "Stochastic volatility model" ? names = ["α", "ρ", "σ", "lnℒ "] :    names = ["μ1", "μ2 ", "σ1", "σ2", "prob", "lnℒ "]
S = 100
covreps = 500
length = 5000
burnin = 1000
verbosity = 100 # show results every X draws
tuning = 1.0

# define the proposal
junk, Σp = mΣ(θnn, covreps, model, nnmodel, nninfo)
proposal(θ) = rand(MvNormal(θ, tuning*Σp))

# define the logL
lnL = θ -> snmobj(θ, θnn, S, model, nnmodel, nninfo)

# run a short chain to improve proposal
# tuning the chain and creating a good proposal may
# need care - this is just an example!
chain = mcmc(θnn, 1000, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)
Σp = cov(chain[:,1:end-2])
acceptance = mean(chain[:,end])
acceptance < 0.2 ? tuning = 0.75 : nothing
acceptance > 0.3 ? tuning = 1.50 : nothing
proposal2(θ) = rand(MvNormal(θ, tuning*Σp))

# final chain using second round proposal
chain = mcmc(θnn, length, lnL, model, nnmodel, nninfo, proposal2, burnin, verbosity)

# get the summary info
acceptance = mean(chain[:,end])
println("acceptance rate: $acceptance")
# compute RMSE
cc = chain[:,1:end-2]
tp = TrueParameters()
chain = Chains(chain[:,1:end-1], names) # convert to Chains type, drop acc. rate
display(chain)
t = TrueParameters()'
println()
printstyled("For comparison, the true parameters are $t", color=:green)
println()
display(plot(chain))
if saveplot
    whichdgp == "Stochastic volatility model" ? savefig("SVchain.png") : savefig("MNchain.png")
end
rmse = mean(sqrt.((cc .- tp').^2))
return acceptance, rmse
end

