############################################################################
#             	Mini bearing design tool

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	November 2020

# Description: 	Given ratios returned by the ML algorithm, get a proper bearing
# design

# Open issues: 	(1) 

############################################################################
import math
import numpy as np

def designBearing(mu2Ratio, gapRatio, T2Ratio, zeta, Tm):

	g 		= 386.4
	inch 		= 1.0
	pi 			= math.pi
	S1 		= 1.017
	SaTm 	= S1/Tm

	# from ASCE Ch. 17
	zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
	BmRef	= [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]

	Bm 		= np.interp(zeta, zetaRef, BmRef)

	mu2 	= SaTm/Bm*mu2Ratio
	moatGap = g*SaTm/Bm*Tm**2*gapRatio
	T2 		= T2Ratio*Tm
	R2 		= T2**2*g/(8*pi**2)

	RDiv 	= 1.1

	Dm 		= g*S1*Tm/(4*pi**2*Bm)

	if moatGap < Dm:
		print("Weird. Moat gap is smaller than expected displacement.")
		
	x 		= Dm

	xy		= 0.01*inch

	muBad 	= True

	while muBad:

		if RDiv > 10.0:
			break

		R1 		= R2/RDiv

		# R2 		= RDiv*param['R1']
		# R3 		= RDiv*param['R1']

		# k0		= param['mu1']/xy
		# a 		= 1/(2*param['R1'])

		b  		= 1/(2*R2)

		kM 		= (2*pi/Tm)**2 * (1/g)

		x1 		= x - (kM*x - mu2)/b

		Wm 		= zeta*(2*pi*kM*x**2)
		# x1 		= (a-b)**(-1/2)*cmath.sqrt(-Wm/4 + (kM - b)*x**2 - (k0 - a)*xy**2)

		# mu2 	= kM*x - b*(x-x1)
		# mu3 	= mu2

		mu1 	= -x1/(2*R1) + mu2

		k0 		= mu1/xy
		a 		= 1/(2*R1)

		ke 		= (mu2.real + b*(x - x1.real))/x
		We 		= (4*(mu2.real - b*x1.real)*x - 
			4*(a-b)*x1.real**2 - 4*(k0 -a)*xy**2)
		zetaE	= We/(2*pi*ke*x**2)
		Te 		= 2*pi/(math.sqrt(ke/(1/g)))

		muList 	= [mu1, mu2, mu2]

		muBad 	= (any(coeff.real < 0 for coeff in muList) or 
			any(np.iscomplex(muList)))

		RDiv 	+= 0.1

	# Abort if there are nonsensical results
	if(any(coeff.real < 0 for coeff in muList) or any(np.iscomplex(muList))):
		# sys.exit('Bearing solver incorrectly output friction coefficients...')

		# invoke the ValueError or TypeError
		muList 	= [math.sqrt(coeff) for coeff in muList]	

	else:
		muList 		= [coeff.real for coeff in muList]
		RList		= [R1, R2, R2]
		mu1 		= mu1.real
		mu2 		= mu2.real

	return(muList, RList, moatGap)


mu2Ratio 	= 0.25
gapRatio 	= 0.0740
T2Ratio 	= 1.19
zeta 		= 0.15
Tm 			= 3.0

(newMu, newR, newGap) = designBearing(mu2Ratio, gapRatio, T2Ratio, zeta, Tm)