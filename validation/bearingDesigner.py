############################################################################
#             	Mini bearing design tool

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	November 2020

# Description: 	Given ratios returned by the ML algorithm, get a proper bearing
# design

# Open issues: 	(1) Using sympy to solve

############################################################################
import math, cmath
import numpy as np
import sympy as sy
sy.init_printing()

mu2Ratio 	= 0.25
gapRatio 	= 0.03
T2Ratio 	= 1.14
zeta 		= 0.10
Tm 			= 3.5

mu1 		= 0.015

# def designBearing(mu1, gapRatio, T2Ratio, zeta, Tm):

g 		= 386.4
inch 	= 1.0
pi 		= math.pi
S1 		= 1.017
SaTm 	= S1/Tm

# from ASCE Ch. 17
zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef	= [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]

Bm 		= np.interp(zeta, zetaRef, BmRef)

hypmu2 	= SaTm/Bm*mu2Ratio
print("If mu2Ratio is specified, mu2 =", hypmu2)

moatGap = g*(SaTm/Bm)*(Tm**2)*gapRatio
T2 		= T2Ratio*Tm
R2 		= T2**2*g/(8*pi**2)

RDiv 	= 1.1

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

	mu2, R1 = sy.symbols('mu2 R1')

	k0		= mu1/xy
	b  		= 1/(2*R2)

	kM 		= (2*pi/Tm)**2 * (1/g)
	Wm 		= zeta*(2*pi*kM*x**2)

	solset = sy.nsolve( [ (mu2 + 1/(2*R2)*(x - 2*R1*(mu2-mu1))) /x - kM,
		4*(mu2 - 1/(2*R2)*(2*R1*(mu2-mu1)))*x - 4*(1/(2*R1) - 1/(2*R2))*(2*R1*(mu2-mu1))**2 - 4*(k0 - 1/(2*R1))*xy**2 - Wm], 
		[mu2, R1], [0.20, 60])

	npsol = np.array(solset).astype(np.float64)

	mu2 = npsol[0].item()
	R1 = npsol[1].item()

	a 		= 1/(2*R1)

    # need way to tie mu1 to something invariable (R2)
	mu1 	=  mu2 - x1/(2*R1)
	# k0 		= mu1/xy

	print("x1, mu1, mu2, R1, R2: ", "%.3f" % x1, "%.3f" % mu1, 
       "%.3f" % mu2, "%.3f" % R1, "%.3f" % R2)
	

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
if(muBad):
	# sys.exit('Bearing solver incorrectly output friction coefficients...')

	# invoke the ValueError or TypeError
	muList 	= [math.sqrt(coeff) for coeff in muList]	

else:
	muList 		= [coeff.real for coeff in muList]
	RList		= [R1, R2, R2]
	mu1 		= mu1.real
	mu2 		= mu2.real

# return(muList, RList, moatGap)

# (newMu, newR, newGap) = designBearing(mu1Guess, gapRatio, T2Ratio, zeta, Tm)