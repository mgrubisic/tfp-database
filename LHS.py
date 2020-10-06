############################################################################
#             	Latin Hypercube sampling

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	August 2020

# Description: 	Script generates input parameter space based on LHS, given bounds

# Open issues: 	(1) Seek nondimensionalized variables

############################################################################

import numpy as np

from smt.sampling_methods import LHS

def generateInputs(num):

	# range of desired inputs
	# currently, this is approx'd to match MCER level (no site mod)
	inputDict 	= {
		'S1' 			: [0.8, 1.3],
		'S1Ampli'		: [1.0, 2.25],
		'Tm' 			: [2.5, 4.0],
		'zetaM' 		: [0.10, 0.20],
		'mu1' 			: [0.01, 0.05],
		'R1' 			: [15, 45],
		'moatAmpli'		: [1.0, 2.0],
		'RI' 			: [0.5, 2.0]
	}

	# create array of limits, then run LHS
	paramNames 		= list(inputDict.keys())									# variable names. IMPORTANT: Ordered by insertion
	paramLimits 	= np.asarray(list(inputDict.values()), dtype=np.float64)	# variable bounds

	sampling 		= LHS(xlimits=paramLimits)

	sets 			= num
	paramSet 		= sampling(sets)											# values sets

	return(paramNames, paramSet)

if __name__ == '__main__':

	names, inputs 		= generateInputs(50)
	print(inputs.shape)