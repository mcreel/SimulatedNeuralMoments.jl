using Statistics
include("CKlib.jl")
function Analyze(chain)
    lb,ub = PriorSupport()
    θtrue = TrueParameters()
    nParams = size(lb,1)
    inci01 = zeros(nParams)
    inci05 = zeros(nParams)
    inci10 = zeros(nParams)
    lower = zeros(nParams)
    upper = zeros(nParams)
    for i = 1:nParams
        lower[i] = quantile(chain[:,i],0.005)
        upper[i] = quantile(chain[:,i],0.995)
        inci01[i] = θtrue[i] >= lower[i] && θtrue[i] <= upper[i]
        lower[i] = quantile(chain[:,i],0.025)
        upper[i] = quantile(chain[:,i],0.975)
        inci05[i] = θtrue[i] >= lower[i] && θtrue[i] <= upper[i]
        lower[i] = quantile(chain[:,i],0.05)
        upper[i] = quantile(chain[:,i],0.95)
        inci10[i] = θtrue[i] >= lower[i] && θtrue[i] <= upper[i]
    end
    return vcat(mean(chain,dims=1)[:], inci01[:], inci05[:], inci10[:])
end   

