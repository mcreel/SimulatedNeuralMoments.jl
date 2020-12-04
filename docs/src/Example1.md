# SNM
This project explores the use of neural nets to reduce the dimension of statistics used for Approximate Bayesian Computing or the method of simulated moments. This leads to more reliable inference: confidence intervals derived from quantiles of samples from the posterior are found to be more reliable when the statistics are filtered through a neural net.

The project allows for creation and training of the neural net, and for calculation of the neural moments, given the trained net. It also provides the large sample indirect likelihood function of the neural moments, which can be used to sample from the posterior. Sampling is done by MCMC, using a very effective proposal that naturally follows from the neural net estimator. The results reported below are a product of three features: the use of neural moments to reduce the dimension of the summary statistics, and the use of the indirect likelihood function as the criterion or distance measure, and effective MCMC sampling due to a well-chosen proposal distribution. The code for these three features is [here](https://github.com/mcreel/SNM/blob/master/src/SNM.jl).

The project allows for Monte Carlo investigation of the performance of estimators and the reliability of confidence intervals obtained from the quantiles samples from the posterior distribution.

The results of the project are reported in the working paper <a href=https://www.barcelonagse.eu/research/working-papers/inference-using-simulated-neural-moments>Inference using simulated neural moments</a>. The code in the WP branch of this archive holds the code for the continuously updating version. The master branches focus on the two-step version and also adds additional examples and the jump diffusion results. The code in the master branch is simpler, performs as well, and is recommended. 

# Worked example
The following is an explanation of how to use the code in the master branch.

1. git clone the project into a directory. Go to that directory, set the appropriate number of Julia threads, given your hardware, e.g. ```export JULIA_NUM_THREADS=10```
2. start Julia, and do ```]activate .``` to set up the dependencies correctly. This will take quite a while the first time you do it, as the project relies on a number of packages.
3. do ```include("RunProject.jl)```  to run a Monte Carlo study of simple example based on a mixture of normals. 

The mixture of normals model (see the file [MNlib.jl](https://github.com/mcreel/SNM/blob/master/examples/MN/MNlib.jl) for details) draws statistics using the function
```
function auxstat(θ, reps)
    n = 1000
    stats = zeros(reps, 15)
    r = 0.0 : 0.1 : 1.0
    μ1, μ2, σ1, σ2, prob = θ
    for i = 1:reps
        d1=randn(n).*σ1 .+ μ1
        d2=randn(n).*(σ1+σ2) .+ (μ1 - μ2) # second component lower mean and higher variance
        ps=rand(n).<prob
        data=zeros(n)
        data[ps].=d1[ps]
        data[.!ps].=d2[.!ps]
        stats[i,:] = vcat(mean(data), std(data), skewness(data), kurtosis(data),
        quantile.(Ref(data),r))
    end
    sqrt(n).*stats
end    

```    

So, there are five parameters, and 15 summary statistics. Samples of 1000 observations are used to compute the statistics. The "true" parameter values we will use to evaluate performance and confidence interval coverage are from
```
function TrueParameters()
    [1.0, 1.0, 0.2, 1.8, 0.4]
end
```    

When we run ```RunProject()```, as above, a Monte Carlo study of 1000 replications of estimation of the model is done. For each replication, confidence intervals for each of the parameters are computed, and we can observe whether or not the true parameters lie in the respective confidence intervals. We obtain output similar to the following results, for 1000 Monte Carlo replications:
![results](https://github.com/mcreel/SNM/blob/master/examples/MN/results.png)

The parameters are estimated with little bias, and good precision, and confidence interval coverages are close to the nominal levels, for each of the 5 parameters.


4. do ```include("examples/MN/EstimateMN.jl")``` to do a single estimation of the mixture of normals model. We can visualize the posterior densities for the parameters, based on a kernel density fit to the final MCMC chain, and the tail quantiles which define a 90% confidence interval, for each of the five parameters of the model:

For the first parameter, the true value is 1.0. The density plot of the posterior is
![MNp1](https://github.com/mcreel/SNM/blob/master/examples/MN/MNp1.png)

For the second parameter, the true value is 1.0. The density plot of the posterior is
![MNp2](https://github.com/mcreel/SNM/blob/master/examples/MN/MNp2.png)

For the third parameter, the true value is 0.2. The density plot of the posterior is
![MNp3](https://github.com/mcreel/SNM/blob/master/examples/MN/MNp3.png)

For the fourth parameter, the true value is 1.8. The density plot of the posterior is
![MNp4](https://github.com/mcreel/SNM/blob/master/examples/MN/MNp4.png)

For the fifth parameter, the true value is 0.4. The density plot of the posterior is
![MNp5](https://github.com/mcreel/SNM/blob/master/examples/MN/MNp5.png)

