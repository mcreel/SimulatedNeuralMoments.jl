using Pkg
Pkg.activate("./")
ENV["JULIA_MPI_BINARY"]="system"; using Pkg; Pkg.build("MPI"; verbose=true) 
