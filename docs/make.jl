using SimulatedNeuralMoments
using Documenter

makedocs(;
    modules = [SimulatedNeuralMoments],
    authors = "Michael Creel <michael.creel@uab.cat> and contributors",
    repo = "https://github.com/mcreel/SimulatedNeuralMoments.jl/blob/{commit}{path}#L{line}",
    sitename = "SimulatedNeuralMoments.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/mcreel/simulatedNeuralMoments.jl",
        assets = String[],
    ),
    pages = [
        "Example: Gaussian Mixture" => "Example1.md",
    ],
)

deploydocs(; repo = "github.com/mcreel/SimulatedNeuralMoments.jl", push_preview = true)
