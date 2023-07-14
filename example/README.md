# example.jl
The main purpose of this example is to show how to use the methods with real data. The example will run one of two models, a stochastic volatility model with 3 parameters, or a mixture of normals model with five parameters. To run the file, I suggest:
1. download the whole SimulatedNeuralMoments package, and go the the example directory.
2. start Julia using julia --proj -t8 (or the appropriate number of threads for your hardware)  , and then instantiate the project to get all needed packages.
3. ```julia include("example.jl); example()```

You will end up with something like the following:

## stochastic volatility model
The true parameters used to generate the data are 
```julia
function TrueParameters()
    [exp(-0.736/2.0), 0.9, 0.363]
end
```
![SVchain](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/example/SVchain.png)
![SVresults](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/example/SVresults.png)

## mixture of normals model
The true parameters used to generate the data are 
```julia
function TrueParameters()
    [1.0, 1.0, 0.2, 1.8, 0.4] # first component N(1,0.2) second component N(0,0.2+1.8), mix prob = 0.4
end    
```
![MNchain](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/example/MNchain.png)
![MNresults](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/example/MNresults.png)



