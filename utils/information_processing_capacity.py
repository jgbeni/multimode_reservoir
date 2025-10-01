import numpy as np
from utils.degdelaysets import single_input_degdelaysets
from scipy.special import legendre # type: ignore
from tqdm import tqdm # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore

# compute normalized legendre polynomial of x with degree d
def legendre_poly(x,d):
    p = legendre(d)
    Px = np.zeros(len(x),float)
    for i in range(d+1):
        Px += p[i]*x**i
    return np.sqrt((d+1)/2)*Px

# compute list with delayed versions of the inputs
# Two: wash out timesteps
def make_delayed_inputs(u,Two):
    del_inputs = []
    T = len(u)-Two
    for i in range(Two):
        del_inputs.append(u[Two-i:Two+T-i])
    return del_inputs

# compute all legendre polynomias from degree 1 to max_deg and from delay 0 to max_del
# del_inputs has to be the output of make_delayed_inputs function
def legendre_delayed(del_inputs,max_deg,max_del):
    len_input = len(del_inputs[0])
    leg_arr = np.zeros((max_deg,max_del+1,len_input),float)
    for d in range(1,max_deg+1):
        for tau in range(max_del+1):
            leg_arr[d-1,tau,:] = legendre_poly(del_inputs[tau],d)
    return leg_arr

# Function to compute the IPC
# X_res: output observables of the reservoir X_res.shape = (T+Two, # observables)
# u: input sequence, must be uniformly random in the interval [-1, 1]
# Two: wash out timesteps
# degdelay: list of all degrees and max_delays to compute
### e.g. degdelay = [[1,20],[2,15]] will compute the capacities of degree 1 up delay 20
### and capacities of delay 2 up to delay 15
def compute_ipc(X_res,u,Two,degdelay,verbose=True):

	# Compute all degree-delay terms
	# and the maximum degre (max_deg) and delay (max_del)
    degdelaysets = single_input_degdelaysets()
    max_deg,max_del = 0,0
    for deg,delay in degdelay:
        max_deg = max(max_deg,deg)
        max_del = max(max_del,delay)
        degdelaysets.make_degdelaysets(deg,delay)
    
    T = len(u)-Two # timesteps of the training set (we just remove the wash out)
    X = X_res[Two::,:] # training observables (remove the wash out)
    
    del_u = make_delayed_inputs(u,Two) # make list with delayed versions of the input
    del_poly = legendre_delayed(del_u,max_deg,max_del) # make 3D array with all legendre polynomials of the input
    
    
    IPC = np.zeros(max_deg,float)
    for deg,delay in degdelay:
        C_values = []
        degdelay_all = degdelaysets.load(deg,delay) #load degree-delay tuples

        if verbose == True:
            print('--- Degree %i ---' %(deg))
            for i in tqdm(range(len(degdelay_all))):
            
                # we compute the product of the legendre polynomials of the delayed inputs
                poly_degdelay = degdelay_all[i]
                Y = np.ones(T,float)
                for j in range(len(poly_degdelay)):
                    ddlay_poly = poly_degdelay[j]
                    d,tau = ddlay_poly[0],ddlay_poly[1]
                    Y *= del_poly[d-1,tau,:]
            
                #we perform the linear regression   
                clf = LinearRegression()
                clf.fit(X,Y)
                Y_pred = clf.predict(X) #predicted values
            
                # we standardize the target and predicted values
                Y_mean,Y_std = np.mean(Y),np.std(Y)
                Y_norm,Y_pred_norm = (Y-Y_mean)/Y_std,(Y_pred-Y_mean)/Y_std
            
                # we compute and store the capacity
                C = max(0,1.-norm_MSE(Y_norm,Y_pred_norm))
                C_values.append(C)
        else:
            for i in range(len(degdelay_all)):
            
                # we compute the product of the legendre polynomials of the delayed inputs
                poly_degdelay = degdelay_all[i]
                Y = np.ones(T,float)
                for j in range(len(poly_degdelay)):
                    ddlay_poly = poly_degdelay[j]
                    d,tau = ddlay_poly[0],ddlay_poly[1]
                    Y *= del_poly[d-1,tau,:]
            
                #we perform the linear regression   
                clf = LinearRegression()
                clf.fit(X,Y)
                Y_pred = clf.predict(X) #predicted values
            
                # we standardize the target and predicted values
                Y_mean,Y_std = np.mean(Y),np.std(Y)
                Y_norm,Y_pred_norm = (Y-Y_mean)/Y_std,(Y_pred-Y_mean)/Y_std

                # we compute and store the capacity
                C = max(0.,1.-norm_MSE(Y_norm,Y_pred_norm))
                C_values.append(C)
        threshold,C_valid = threshold_compute(C_values) # we compute the threshold to remove the noise
        
        IPC[deg-1] = np.sum(C_valid) # we sum the values above the threshold
        if verbose:
            print('threshold =',threshold)
            print('*** IPC = %.2f ***' %(IPC[deg-1]))
    return IPC

# compute normalized MSE
def norm_MSE(Y,Y_hat):
    return np.mean((Y-Y_hat)**2)/np.mean(Y**2)

# compute threshold
def threshold_compute(C_values):
	C_values = np.array(C_values,float)
	if np.median(C_values)<0.2:
		C_std_last = np.std(C_values[-int(0.5*len(C_values))::])
		C_p = np.percentile(C_values, 95)
		threshold = min(0.2,C_p + max(0.04,4*C_std_last))
	else:
		threshold = 0.2
	C_valid = C_values[C_values>=threshold]
	return threshold,C_valid