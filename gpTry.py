############################################################################
#             	Gaussian process trials

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	October 2020

# Description:  Early trial of kriging for isolation data

############################################################################

import pandas as pd
import numpy as np
import GPy
import matplotlib.pyplot as plt

plt.close('all')

############################################################################
#             	Utilities
############################################################################

def reshapeArray(arr):
	arr.shape = (len(arr), 1)
	return(arr)

############################################################################
#             	Load data
############################################################################

dataDir 	= '../pastRuns/random200withTfb.csv'

isolData 	= pd.read_csv(dataDir, sep=',')

noImpact	= isolData[isolData.impacted == 0].copy()

# defined Pi groups
TfbRatio 	= noImpact['Tfb']/noImpact['Tm']
mu2Ratio 	= noImpact['mu2']/noImpact['GMSTm']
gapRatio 	= noImpact['moatGap']/(noImpact['GMSTm']*noImpact['Tm']**2)
T2Ratio 	= noImpact['GMST2']/noImpact['GMSTm']

# Remaining pi groups: RI, zetaM, S1Ampli

TfbRatio 	= reshapeArray(TfbRatio.to_numpy())
mu2Ratio 	= reshapeArray(mu2Ratio.to_numpy())
gapRatio 	= reshapeArray(gapRatio.to_numpy())
T2Ratio  	= reshapeArray(T2Ratio.to_numpy())
Ry 			= reshapeArray(noImpact['RI'].to_numpy())
zeta 		= reshapeArray(noImpact['zetaM'].to_numpy())
A_S1		= reshapeArray(noImpact['S1Ampli'].to_numpy())

# outputs are displacements and drifts

drift1 		= noImpact['driftMax1'].to_numpy()
drift2 		= noImpact['driftMax2'].to_numpy()
drift3 		= noImpact['driftMax3'].to_numpy()
maxDrift 	= reshapeArray(np.maximum(drift1, drift2, drift3))
maxDisp		= reshapeArray(noImpact['maxDisplacement'].to_numpy())

############################################################################
#             	Kriging stuff
############################################################################

# define covariance kernel (Gaussian kernel, rbf or sq exp)
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

# model is made from X, Y, and kernel
m = GPy.models.GPRegression(mu2Ratio, maxDisp, kernel)
print(m)

fig = m.plot()

m.optimize()
print(m)

fig = m.plot()