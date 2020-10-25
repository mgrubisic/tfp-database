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
from IPython.display import display

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

# do subsetting of data here (impact, not impact)
dataSubset	= isolData.copy()

# defined Pi groups
TfbRatio 	= dataSubset['Tfb']/dataSubset['Tm']
mu2Ratio 	= dataSubset['mu2']/dataSubset['GMSTm']
gapRatio 	= dataSubset['moatGap']/(dataSubset['GMSTm']*dataSubset['Tm']**2)
T2Ratio 	= dataSubset['GMST2']/dataSubset['GMSTm']

# Remaining pi groups: RI, zetaM, S1Ampli

TfbRatio 	= reshapeArray(TfbRatio.to_numpy())
mu2Ratio 	= reshapeArray(mu2Ratio.to_numpy())
gapRatio 	= reshapeArray(gapRatio.to_numpy())
T2Ratio  	= reshapeArray(T2Ratio.to_numpy())
Ry 			= reshapeArray(dataSubset['RI'].to_numpy())
zeta 		= reshapeArray(dataSubset['zetaM'].to_numpy())
A_S1		= reshapeArray(dataSubset['S1Ampli'].to_numpy())

# outputs are displacements and drifts

drift1 		= dataSubset['driftMax1'].to_numpy()
drift2 		= dataSubset['driftMax2'].to_numpy()
drift3 		= dataSubset['driftMax3'].to_numpy()
maxDrift 	= reshapeArray(np.maximum(drift1, drift2, drift3))
maxDisp		= reshapeArray(dataSubset['maxDisplacement'].to_numpy())

allPis      = np.concatenate([TfbRatio, mu2Ratio, gapRatio, T2Ratio,
                              Ry, zeta, A_S1], axis = 1)

############################################################################
#             	Kriging stuff
############################################################################

# # define covariance kernel (Gaussian kernel, rbf or sq exp)
# kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

# # kernel = GPy.kern.Matern52(7,ARD=True) + GPy.kern.White(7)

# # model is made from X, Y, and kernel
# m = GPy.models.GPRegression(mu2Ratio, maxDisp, kernel)
# print(m)

# fig = m.plot()

# m.optimize()
# print(m)

# fig = m.plot()

############################################################################
#             	Classification trials
############################################################################

X	= np.concatenate([gapRatio, mu2Ratio], axis = 1)
Y 	= reshapeArray(dataSubset['impacted'].to_numpy())

# assumes RBF kernel, EP inference
m 	= GPy.models.GPClassification(X,Y)

m.plot()
plt.ylabel('$\mu_2$ ratio');plt.xlabel('Gap ratio')

# optimize
print(m, '\n')

for i in range(5):
	#first runs EP and then optimizes the kernel parameters
	m.optimize('bfgs', max_iters=100) 

	print('iteration: ', i)
	print(m)
	print(' ')

fig = m.plot()
# plt.ylabel('$\mu_2$ ratio');plt.xlabel('Gap ratio')
# plt.legend()

display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))