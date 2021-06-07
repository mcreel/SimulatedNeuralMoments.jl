# These are the candidate auxiliary statistics for ABC estimation of
# the simple DSGE model of Creel and Kristensen (2013)
using Statistics, LinearAlgebra

function bad_data(data)
    any(isnan.(data)) || any(isinf.(data)) || any(std(data,dims=1) .==0.0) || any(data .< 0.0)
end

function auxstat(θ, reps)
    data = dgp(θ)
    Z = auxstat(data)
    stats = zeros(reps, size(Z,1))
    stats[1,:] = Z
    for rep = 2:reps
        data = dgp(θ)
        stats[rep, :] = auxstat(data)
    end
    stats
end    

function auxstat(data)
    # check for nan, inf, no variation, or negative, all are reasons to reject
    if bad_data(data)
        return zeros(39)
    else    
        # known parameters
        α = 0.33
        δ = 0.025
        # recover capital (see notes)
        hours = data[:,3]
        intrate = data[:,4]
        wages = data[:,5]
        capital = α /(1.0-α )*hours.*wages./intrate
        # treat all variables
        data = [data capital] 
        logdata = log.(data)
        lagdata = data[1:end-1,:]
        laglogdata = logdata[1:end-1]
        data = data[2:end,:]
        logdata = logdata[2:end,:]
        # break out variables
        output = data[:,1]
        cons = data[:,2]
        hours = data[:,3]
        intrate = data[:,4]
        wages = data[:,5]
        capital = data[:,6]
        # logs
        logoutput = logdata[:,1];   # output
        logcons = logdata[:,2];     # consumption
        loghours = logdata[:,3];    # hours
        logintrate = logdata[:,4];  # intrate
        logwages = logdata[:,5];    # wages
        logcapital = logdata[:,6]
        # lags
        lagoutput = lagdata[:,1];   # output
        lagcons = lagdata[:,2];     # consumption
        laghours = lagdata[:,3];    # hours
        lagintrate = lagdata[:,4];  # intrate
        lagwages = lagdata[:,5];    # wages
        lagcapital = lagdata[:,6]

        # rho1, sig1
        e = logoutput-α*logcapital-(1.0-α)*loghours 
        y = e[2:end]
        x = e[1:end-1]
        rho1 = cor(x,y)
        u = y-x*rho1
        sig1 = sqrt(u'*u/size(u,1))
        Z = vcat(rho1, sig1)
        # gam, rho2, sig2 (1/MRS=wage)
        x = [ones(size(logcons,1)) logcons]
        y = logwages
        b = x\y
        e = y-x*b
        y = e[2:end]
        x = e[1:end-1]
        rho2 = cor(y,x)
        u = y-x*rho2
        sig2 = sqrt(u'*u/size(u,1))
        Z = vcat(Z, b, rho2, sig2)
        # standard devs. and correlations
        m = mean(logdata, dims=1)
        s = std(logdata, dims=1)
        d = (logdata .- m) ./s # keep means and std. devs., the VAR uses standardized and normalized   
        # AR(1)
        maxlag = 1
        y = d[2:end,:]
        x = d[1:end-1,:]
        n = size(y,1)
        rhos = zeros(6)
        es = zeros(n,6)
        for i = 1:6
            rho = x[:,i]\y[:,i]
            rhos[i] = rho;
            es[:,i] = y[:,i]-rho*x[:,i]
        end        
        varv = vech(cov(es)) # AR(1) error covariance elements 
        Z = vcat(Z, m[:], s[:], varv)
    end
    Z
end

function vech(x)
    k = size(x,1)
    a = zeros(Int((k^2-k)/2 + k))
    m = 1
    for i = 1:k
        for j = 1:i
            a[m] = x[i,j]
            m += 1
        end
    end
    a
end


