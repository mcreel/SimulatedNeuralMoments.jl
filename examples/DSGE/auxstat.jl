# These are the candidate auxiliary statistics for ABC estimation of
# the simple DSGE model of Creel and Kristensen (2013)
using Econometrics, Statistics

function auxstat(data)
    # check for nan, inf, no variation, or negative, all are reasons to reject
    function bad_data(data)
        any(isnan.(data)) || any(isinf.(data)) || any(std(data,dims=1) .==0.0) || any(data .< 0.0)
    end
    if bad_data(data)
        return false
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
        lagdata = lags(data,1)
        laglogdata = lags(logdata,1)
        # line up in time
        data = data[2:end,:]
        lagdata = lagdata[2:end,:]
        logdata = logdata[2:end,:]
        laglogdata = laglogdata[2:end,:]
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

        n = size(data,1)
        # rho1, sig1
        x = [ones(n) logcapital loghours]
        y = logoutput
        b = x\y
        e = y-x*b
        junk = [e lag(e,1)][2:end,:]
        y = junk[:,1]
        x = junk[:,2]
        rho1 = cor(x,y)
        e = y-x*rho1
        sig1 = e'*e/n
        Z = vcat(1.0-b[3], rho1, sig1)

        # gam, rho2, sig2 (1/MRS=wage)
        x = [ones(n,1) logcons]
        y = logwages
        b = x\y
        e = y-x*b
        junk = [e lag(e,1)][2:end,:]
        y = junk[:,1]
        x = junk[:,2]
        rho2 = cor(y,x)
        e = y-x*rho2
        sig2 = e'*e/n
        Z = vcat(Z, rho2, sig2)

        # standard devs. and correlations
        data = data[:,1:5]
        data, m, s = stnorm(data) # keep means and std. devs., the VAR uses standardized and normalized

        # AR(1)
        maxlag = 1
        data = [data lags(data, 1)] # add lags
        data = data[2:end,:] # drop rows with missing
        y = data[:,1:5]
        x = data[:,6:end]
        n = size(y,1)
        rhos = zeros(5,1)
        es = zeros(n,5)
        for i = 1:5
 HERE           
            [rho, junk, e] = ols(y(:,i),x(:,i));
            rhos(i,:) = rho;
            es(:,i) = e;
        end        
        varv = vech(cov(es)); % AR(1) error covariance elements 
        # ratios
        s1 = mean(cons./output);
        s2 = mean(intrate./wages);
        s3 = mean(cons./hours);
        Z = [Z; m(:); s(:); rhos(:); varv(:); s1; s2; s3];
        Z = real(Z);
        if check_bad_data(Z)
            bad_data = true;
        end
=#
    end
      #      Z = -1000*ones(7 + 5 + 5  + 5 + 15 + 3,1);
    @show Z
end


