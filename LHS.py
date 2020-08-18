############################################################################
#             	Latin Hypercube sampling

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	August 2020

# Description: 	Script generates input parameter space based on LHS, given bounds

# Open issues: 	(1) Potentially switch to new distributions
# 				(2) Use a list instead of manual arrays
# 				(3) Seek nondimensionalized variables

############################################################################

import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

def generateInputs(num):

	# range of desired inputs
	S1Bounds 		= [0.8, 1.3]
	S1AmpliBounds	= [0.75, 1.25]
	T1Bounds 		= [2.5, 4.0]
	zetaBounds 		= [0.10, 0.20]
	mu1Bounds 		= [0.01, 0.05]
	R1Bounds 		= [15, 45]
	moatAmpliBounds	= [1.0, 1.25]
	RIBounds 		= [0.5, 2.5]

	# create array of limits, then run LHS
	paramLimits = np.array([S1Bounds, S1AmpliBounds, T1Bounds, zetaBounds, mu1Bounds, R1Bounds, moatAmpliBounds, RIBounds])
	sampling = LHS(xlimits=paramLimits)

	# create the inputs
	sets = num
	x = sampling(sets)

	return(x)

if __name__ == '__main__':

	inputs 		= generateInputs(50)

	print(inputs.shape)

	plt.close('all')
	plt.figure()
	plt.plot(inputs[:, 2], inputs[:, 3], "o")
	plt.xlabel("mu1")
	plt.ylabel("R1")
	plt.show()