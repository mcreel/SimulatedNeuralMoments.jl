# SNM examples
The main purpose of these examples is to show how to use the methods with real data. The examples will run one of two models, a stochastic volatility model with 3 parameters, or a mixture of normals model with five parameters.

There is a video that explains the example at [example](https://youtu.be/Ps-gl8Hz-20).

To run the example yourself, I suggest:
1. download the whole SimulatedNeuralMoments package using git clone or as a zip file
2. go the example directory
3. start Julia using julia --proj -t8 (or the appropriate number of threads for your hardware), and then instantiate the project to get all needed packages.
    ```using Pkg; Pkg.instantiate()```
4. run the example by doing  ```include("Example.jl")```

You will end up with something like the following:

## SV: stochastic volatility model
![SVchain](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/SVchain.png)
![SVresults](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/SVresults.png)

## MN: mixture of normals model
![MNchain](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/MNchain.png)
![MNresults](https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/main/examples/MNresults.png)



