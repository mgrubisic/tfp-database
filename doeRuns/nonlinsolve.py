import numpy as np
import math, cmath
import sympy as sy

gapRatio 	= 1.15
TmRatio 	= 7.4
T2Ratio 	= 1.05
zeta  = 0.15
mu1In 		= 0.03

g 		= 386.4
inch 	= 1.0
pi 		= math.pi
S1 		= 1.017

Ss 		= 2.281
Ts 		= S1/Ss
Tm 		= TmRatio*Ts

SaTm 	= S1/Tm

# from ASCE Ch. 17
zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BmRef	= [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]

Bm 		= np.interp(zeta, zetaRef, BmRef)


moatGap = math.ceil(g*(SaTm/Bm)*(Tm**2)*gapRatio/(4*pi**2))

T2Bad = True

while T2Bad:
	
	if T2Ratio > 1.25:
		break
	
	T2 		= T2Ratio*Tm
	R2 		= T2**2*g/(8*pi**2)
	
	Dm 		= g*S1*Tm/(4*pi**2*Bm)
	
	# Bearing design starts here
	x 		= Dm
	# yield displ
	xy		= 0.01*inch
	
	
	# muguess will start from the LHS and grow outward +/- 0.03
	a1 = np.arange(0.01, 0.04, 0.01)
	a1 = np.repeat(a1,2)
	a2 = [(-1)**i for i in range(len(a1))]
	muTries = mu1In + np.multiply(a1,a2)
	muTries = muTries[muTries > 0]
	
	# start with the initial read in
	muTries = np.insert(muTries, 0, mu1In, axis = 0)
	for mu1Try in muTries:
		try:
			mu1 = mu1Try
			k0		= mu1/xy
	
			b  		= 1/(2*R2)
		
			kM 		= (2*pi/Tm)**2 * (1/g)
			Wm 		= zeta*(2*pi*kM*x**2)
			
			mu2, R1 = sy.symbols('mu2 R1')
			
			# initial guesses will try to readjust for 10 attempts
			attemptsGuess = 0
			attemptsMax = 10
			mu2Guess = 0.05
			R1Guess = R2/2
			
			solset = None
			
			while attemptsGuess < attemptsMax:
				try:
					solset = sy.nsolve( [ (mu2 + 1/(2*R2)*(x - 2*R1*(mu2-mu1))) /x - kM,
						4*(mu2 - 1/(2*R2)*(2*R1*(mu2-mu1)))*x \
							- 4*(1/(2*R1) - 1/(2*R2))*(2*R1*(mu2-mu1))**2 \
								- 4*(k0 - 1/(2*R1))*xy**2 - Wm],
						[mu2, R1], [mu2Guess, R1Guess])
				except ValueError:
					attemptsGuess += 1
					#print("Bad solve, try new starting guess for mu2...")
					mu2Guess += 0.01
					continue
				else:
					npsol = np.array(solset).astype(np.float64)
					mu2 	= npsol[0].item()
					R1 		= npsol[1].item()
					mu3 	= mu2
					break
			# check to see if mu2 is positive
			testMu2 = math.sqrt(mu2)
		except (ValueError, TypeError) as e:
			#print("Bad solve, trying a new mu1...")
			continue
		else:
			break
	
	
	# mu2 	= kM*x - b*(x-x1)
	
	# catch cases where we were unable to solve using mu1 guesses
	try:
		a 		= 1/(2*R1)
		x1 		= (a-b)**(-1/2)*cmath.sqrt(-Wm/4 + (kM - b)*x**2 - (k0 - a)*xy**2)
	except TypeError:
		T2Bad = True
		T2Ratio = T2Ratio + 0.01
		continue
	
	mu1 	= mu2 - x1/(2*R1)
	
	ke 		= (mu2.real + b*(x - x1.real))/x
	We 		= (4*(mu2.real - b*x1.real)*x - 
		4*(a-b)*x1.real**2 - 4*(k0 -a)*xy**2)
	zetaE	= We/(2*pi*ke*x**2)
	Te 		= 2*pi/(cmath.sqrt(ke/(1/g)))
	
	muList 	= [mu1.real, mu2.real, mu3.real]
	
	# conditions for a sensible bearing
# 	T2Bad = (mu2.real > 2.5*mu1.real) or (mu2.real < mu1.real) \
# 		 or (R2 < 2*R1) or (R1 < 10)
	T2Bad = (mu2.real > 0.13) or (mu2.real < 0.05) or (mu2.real < mu1.real) \
		 or (R2 < 2*R1) or (R1 < 10)
	T2Ratio = T2Ratio + 0.01
		
RList 	= [R1, R2, R2]

print('Selected friction coeffs: ', muList)
print('Selected curvature radii: ', RList)
print('Moat gap:', moatGap)
print('Damping: ', zeta)
print('Tm: ', Tm)
print('Final T2 ratio: ', T2Ratio)
print('T2: ', T2)