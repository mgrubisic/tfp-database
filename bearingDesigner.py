############################################################################
#             	Mini bearing design tool

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	November 2020

# Description: 	Given ratios returned by the ML algorithm, get a proper bearing
# design

# Open issues: 	(1) Overconstrained? Yes: mu2 has to be adjusted
#				(2) Even if mu2 is adjusted, previous design algorithm struggles
#				(3) Current solution: revert to old design by guessing mu1

############################################################################
import math, cmath
import numpy as np

mu2Ratio 	= 0.25
gapRatio 	= 0.0740
T2Ratio 	= 1.19
zeta 		= 0.15
Tm 			= 3.5

mu1Guess 	= 0.03

def designBearing(mu1, gapRatio, T2Ratio, zeta, Tm):

	g 		= 386.4
	inch 	= 1.0
	pi 		= math.pi
	S1 		= 1.017
	SaTm 	= S1/Tm

	# from ASCE Ch. 17
	zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
	BmRef	= [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]

	Bm 		= np.interp(zeta, zetaRef, BmRef)
    
	print(Bm)

	hypmu2 	= SaTm/Bm*mu2Ratio
	print("If mu2Ratio is specified, mu2 =", hypmu2)

	moatGap = g*(SaTm/Bm)*(Tm**2)*gapRatio
	T2 		= T2Ratio*Tm
	R2 		= T2**2*g/(8*pi**2)

	RDiv 	= 2.0

	Dm 		= g*S1*Tm/(4*pi**2*Bm)
	print("Expected displacement =", Dm)

	if moatGap < Dm:
		print("Weird. Moat gap is smaller than expected displacement.")
		
	x 		= Dm

	xy		= 0.01*inch

	muBad 	= True

	while muBad:

		if RDiv > 10.0:
			print("Could not find a design within good R ratios.")
			break

		R1 		= R2/RDiv

		# R2 		= RDiv*param['R1']
		# R3 		= RDiv*param['R1']

		k0		= mu1/xy
		# a 		= 1/(2*param['R1'])

		a 		= 1/(2*R1)
		b  		= 1/(2*R2)

		kM 		= (2*pi/Tm)**2 * (1/g)
		Wm 		= zeta*(2*pi*kM*x**2)

		# x1 		= x + (mu2 - kM*x)/b
		
		x1 		= (a-b)**(-1/2)*cmath.sqrt(-Wm/4 + (kM - b)*x**2 - (k0 - a)*xy**2)

		# k0 		= -1/(xy**2)*(x1**2*(a-b) + Wm/4 - (kM-b)*x**2) + a
		# mu1 	= k0*xy

		# mu2 	= mu1 + x1/(2*R1)

		mu2 	= kM*x - b*(x-x1)
		# mu3 	= mu2

		mu1 	= -x1/(2*R1) + mu2
		# k0 		= mu1/xy
		

		ke 		= (mu2.real + b*(x - x1.real))/x
		We 		= (4*(mu2.real - b*x1.real)*x - 
			4*(a-b)*x1.real**2 - 4*(k0 -a)*xy**2)
		zetaE	= We/(2*pi*ke*x**2)
		Te 		= 2*pi/(cmath.sqrt(ke/(1/g)))

		muList 	= [mu1, mu2, mu2]

		muBad 	= (any(coeff.real < 0 for coeff in muList) or 
			any(np.iscomplex(muList))) or (mu1 > mu2)

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

(newMu, newR, newGap) = designBearing(mu1Guess, gapRatio, T2Ratio, zeta, Tm)