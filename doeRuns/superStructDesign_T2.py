############################################################################
#             	Design algorithm (DOE)

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	June 2021

# Description: 	Script designs TFP bearing given site parameters and desired stiffnesses.
# 				Specifications for bearings: Tm, mu1, R2. Design starts from zeta
# and solves for R1, mu2
# 				Script also designs superstructure following ASCE 7-16 Ch.12 & 17 provisions
#				Inputs rely on dictionary calling
# 				Adapted for DOE study
#               Constraints are placed on sensible design (mu2 <= 3*mu1), at the cost of not specifying damping

# Open issues: 	(1) Rework beam and column selection using repeatable functions

############################################################################

# import libraries
import numpy as np
import math, cmath
import pandas as pd
import sympy as sy
sy.init_printing()

############################################################################
#              Searcher utilities
############################################################################

def getProp(shape):
	Zx 		= float(shape.iloc[0]['Zx'])
	Ag 		= float(shape.iloc[0]['A'])
	Ix 		= float(shape.iloc[0]['Ix'])
	bf 		= float(shape.iloc[0]['bf'])
	tf 		= float(shape.iloc[0]['tf'])
	return(Ag, bf, tf, Ix, Zx)

def calcStrength(shape, VGrav):
	Zx 		= float(shape.iloc[0]['Zx'])

	Mn 			= Zx*Fy
	Mpr 		= Mn*Ry*Cpr
	Vpr 		= 2*Mpr/LBay
	beamVGrav 	= 2*VGrav

	return(Mn, Mpr, Vpr, beamVGrav)

def design():

	############################################################################
	#              Bearing Design
	############################################################################

	# prepare global variables for calculation functions
	global Fy, Ry, Cpr, LBay

	# units: in, kip, s
	# dimensions
	inch 	= 1.0
	in2 	= inch*inch
	in4 	= inch*inch*inch*inch
	ft 		= 12.0*inch
	sec 	= 1
	g 		= 386.4*inch/(sec**2)
	pi 		= math.pi
	kip 	= 1
	ksi 	= kip/(inch**2)

	# TFP Algorithm: Becker & Mahin
	bearingParams = pd.read_csv('./inputs/bearingInput_T2.csv', 
		index_col=None, header=0)

	# param is dictionary of all inputs. call with param['whatYouWant']
	param 	= dict(zip(bearingParams.variable, bearingParams.value))

	# Building params, hardcoded
	# Mostly constant, so not varying
	Ws 		= 2227.5*kip
	W 		= 3037.5*kip
	nFrames = 2
	Tfb 	= 0.8*sec

	# from ASCE Ch. 17
	zetaRef = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
	BmRef	= [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0]
	
	zetaGuess = param['zetaM']

	Ts 				= param['S1']/param['Ss']
	param['Tm'] 	= param['TmRatio']*Ts
	print('Tm: ', param['Tm'])

	SaTm 	= param['S1']/param['Tm']

	T2 		= param['T2Ratio']*param['Tm']
	print('T2: ', T2)
	R2 		= T2**2*g/(8*pi**2)
	R3 		= R2
	
	zetaBad = True
	
	while zetaBad:
		
		if zetaGuess > 0.20:
			break
		
		# moat gap from GPML's ratio
		Bm 		= np.interp(zetaGuess, zetaRef, BmRef)
		moatGap = math.ceil(g*(SaTm/Bm)*(param['Tm']**2)*param['gapRatio']/(4*pi**2))
		Dm 		= g*param['S1']*param['Tm']/(4*pi**2*Bm)
			
		# Bearing design starts here
		x 		= Dm
		# yield displ
		xy		= 0.01*inch
	
	
		# muguess will start from the LHS and grow outward +/- 0.03
		a1 = np.arange(0.01, 0.04, 0.01)
		a1 = np.repeat(a1,2)
		a2 = [(-1)**i for i in range(len(a1))]
		muTries = param['mu1']+np.multiply(a1,a2)
		muTries = muTries[muTries > 0]
		
		# start with the initial read in
		muTries = np.insert(muTries, 0, param['mu1'], axis = 0)
		for mu1Try in muTries:
			try:
				mu1 = mu1Try
				k0		= mu1/xy
	
				b  		= 1/(2*R2)
			
				kM 		= (2*pi/param['Tm'])**2 * (1/g)
				Wm 		= zetaGuess*(2*pi*kM*x**2)
				
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
				# print("Bad solve, trying a new mu1...")
				continue
			else:
				break
	
	
		# mu2 	= kM*x - b*(x-x1)
		
		# catch cases where we were unable to solve using mu1 guesses
		try:
			a 		= 1/(2*R1)
			x1 		= (a-b)**(-1/2)*cmath.sqrt(-Wm/4 + (kM - b)*x**2 - (k0 - a)*xy**2)
		except TypeError:
			zetaBad = True
			zetaGuess = zetaGuess + 0.01
			continue
		mu1 	= mu2 - x1/(2*R1)
		
	
		ke 		= (mu2.real + b*(x - x1.real))/x
		We 		= (4*(mu2.real - b*x1.real)*x - 
			4*(a-b)*x1.real**2 - 4*(k0 -a)*xy**2)
		zetaE	= We/(2*pi*ke*x**2)
		Te 		= 2*pi/(cmath.sqrt(ke/(1/g)))
	
		muList 	= [mu1, mu2, mu3]
		
		# conditions for a sensible bearing
		zetaBad = (mu2.real > 2.5*mu1.real) or (mu2.real < mu1.real) \
			 or (R2 < 2*R1) or (R1 < 10)
		zetaGuess = zetaGuess + 0.01

	print('Final damping: ', zetaGuess)

	# Invoke error if all reasonable bearings have been tried without success
	if(any(coeff.real < 0 for coeff in muList) or any(np.iscomplex(muList))):
		# sys.exit('Bearing solver incorrectly output friction coefficients...')

		# invoke the ValueError or TypeError
		muList 	= [math.sqrt(coeff) for coeff in muList]	

	else:
		muList 		= [coeff.real for coeff in muList]
		RList		= [R1, R2, R3]
		mu1 		= mu1.real
		mu2 		= mu2.real
		mu3 		= mu3.real

	############################################################################
	#              ASCE 7-16: Story forces
	############################################################################

	wx 		= np.array([810.0*kip, 810.0*kip, 607.5*kip]) 			# Floor seismic weights
	hx 		= np.array([13.0*ft, 26.0*ft, 39.0*ft])					# Floor elevations
	hCol 	= np.array([13.0*ft, 13.0*ft, 7.5*ft])					# Column moment arm heights
	hsx 	= np.array([13.0*ft, 13.0*ft, 13.0*ft])					# Column heights
	wLoad 	= np.array([2.72*kip/ft, 2.72*kip/ft, 1.94*kip/ft]) 	# Floor line loads

	Vb 		= (Dm * ke * Ws)/nFrames
	Vst 	= (Vb*(Ws/W)**(1 - 2.5*zetaE))
	Vs 		= (Vst/param['RI'])
	F1 		= (Vb - Vst)/param['RI']

	k 		= 14*zetaE*Tfb

	hxk 	= hx**k

	CvNum 	= wx*hxk
	CvDen 	= np.sum(CvNum)

	Cvx 	= CvNum/CvDen

	Fx 		= Cvx*Vs

	############################################################################
	#              ASCE 7-16: Steel moment frame design
	############################################################################

	nFloor  	= len(wx)
	nBay 		= 3
	LBay 		= 30*ft
	Cd 			= param['RI']

	thetaMax 	= 0.015			# ASCE Table 12.12-1 drift limits
	delx 		= thetaMax*hsx
	delxe 		= delx*(1/Cd)	# assumes Ie = 1.0

	# element lateral force
	Q 			= np.empty(nFloor)
	Q[-1] 		= Fx[-1]

	for i in range(nFloor-2, -1, -1):
		Q[i] = Fx[i] + Q[i+1]

	q 			= Q/nBay

	# beam-column relationships
	alpha 		= 0.8			# strain ratio
	dcdb 		= 0.5			# depth ratio
	beta 		= dcdb/alpha

	# required I
	E 			= 29000*ksi
	Ib 			= q*hCol**2/(12*delxe*E)*(hCol/beta + LBay)								# story beams
	Ib[-1] 		= q[-1]*hsx[-1]**2/(12*delxe[-1]*E)*(hsx[-1]/(2*beta) + LBay)			# roof beams, using hsx since flexibility method assumes h = full column
	Ic 			= Ib*beta

	# required Z
	Fy 				= 50*ksi
	Fu 				= 65*ksi
	MGrav 			= wLoad*LBay**2/12
	VGravStory 		= max(wLoad*LBay/2)
	VGravRoof 		= wLoad[-1]*LBay/2
	MEq 			= q*hCol/2
	Mu 				= MEq + MGrav
	Zb 				= Mu/(0.9*Fy)

	############################################################################
	#              ASCE 7-16: Import shapes
	############################################################################

	IBeamReq 		= Ib.max()
	IColReq 		= Ic.max()
	ZBeamReq 		= Zb.max()

	IBeamRoofReq 	= Ib[-1]
	ZBeamRoofReq 	= Zb[-1]

	beamShapes 		= pd.read_csv('./inputs/beamShapes.csv', index_col=None, header=0)
	sortedBeams 	= beamShapes.sort_values(by=['Ix'])

	colShapes 		= pd.read_csv('./inputs/colShapes.csv', index_col=None, header=0)
	sortedCols 		= colShapes.sort_values(by=['Ix'])

	############################################################################
	#              ASCE 7-16: Capacity design
	############################################################################

	############################################################################
	#              Floor beams

	# select beams that qualify Ix requirement
	qualifiedIx 	= sortedBeams[sortedBeams['Ix'] > IBeamReq]			# eliminate all shapes with insufficient Ix
	sortedWeight 	= qualifiedIx.sort_values(by=['W'])					# select lightest from list		
	selectedBeam 	= sortedWeight.iloc[:1]

	############################################################################
	#              Beam checks

	# Zx check
	(beamAg, beambf, beamtf, beamIx, beamZx) = getProp(selectedBeam)

	if(beamZx < ZBeamReq):
		# print("The beam does not meet Zx requirement. Reselecting...")
		qualifiedZx 		= qualifiedIx[qualifiedIx['Zx'] > ZBeamReq]				# narrow list further down to only sufficient Zx
		sortedWeight 		= qualifiedZx.sort_values(by=['W'])						# select lightest from list
		selectedBeam 		= sortedWeight.iloc[:1]
		(beamAg, beambf, beamtf, beamIx, beamZx) = getProp(selectedBeam)

	Ry 				= 1.1
	Cpr 			= (Fy + Fu)/(2*Fy)

	(beamMn, beamMpr, beamVpr, beamVGrav) 	= calcStrength(selectedBeam, VGravStory)

	# PH location check
	phVGrav 		= wLoad[:-1]*LBay/2
	phVBeam 		= 2*beamMpr/(0.8*LBay)											# 0.9LBay for plastic hinge length
	phLocation 		= phVBeam > phVGrav

	if False in phLocation:
		# print('Detected plastic hinge away from ends. Reselecting...')
		ZBeamPHReq 			= max(wLoad*LBay**2/(4*Fy*Ry*Cpr))
		qualifiedZx 		= qualifiedIx[qualifiedIx['Zx'] > ZBeamPHReq]				# narrow list further down to only sufficient Zx
		sortedWeight 		= qualifiedZx.sort_values(by=['W'])							# select lightest from list
		selectedBeam 		= sortedWeight.iloc[:1]
		(beamAg, beambf, beamtf, beamIx, beamZx) 	= getProp(selectedBeam)
		(beamMn, beamMpr, beamVpr, beamVGrav) 		= calcStrength(selectedBeam, VGravStory)

	# beam shear
	beamAweb 		= beamAg - 2*(beamtf*beambf)
	beamVn 			= 0.9*beamAweb*0.6*Fy

	beamShearFail 	= beamVn < beamVpr

	while beamShearFail:
		# print('Beam shear check failed. Reselecting...')
		AgReq 				= 2*beamVpr/(0.9*0.6*Fy)								# Assume web is half of gross area

		if 'qualifiedZx' in dir():
			qualifiedAg			= qualifiedZx[qualifiedZx['A'] > AgReq]				# narrow Zx list down to sufficient Ag
		else:
			qualifiedAg 		= qualifiedIx[qualifiedIx['A'] > AgReq]				# if Zx check wasn't done previously, use Ix list

		sortedWeight 		= qualifiedAg.sort_values(by=['W'])						# select lightest from list
		selectedBeam 		= sortedWeight.iloc[:1]

		# recheck beam shear
		(beamAg, beambf, beamtf, beamIx, beamZx) = getProp(selectedBeam)
		
		beamAweb 		= beamAg - 2*(beamtf*beambf)
		beamVn 			= 0.9*beamAweb*0.6*Fy

		(beamMn, beamMpr, beamVpr, beamVGrav) 	= calcStrength(selectedBeam, VGravStory)
		
		beamShearFail 	= beamVn < beamVpr

	############################################################################
	#              Roof beams

	# select beams that qualify Ix requirement
	qualifiedIx 		= sortedBeams[sortedBeams['Ix'] > IBeamRoofReq]			# eliminate all shapes with insufficient Ix
	sortedWeight 		= qualifiedIx.sort_values(by=['W'])						# select lightest from list		
	selectedRoofBeam 	= sortedWeight.iloc[:1]

	############################################################################
	#              Roof beam checks

	# Zx check
	(roofBeamAg, roofBeambf, roofBeamtf, roofBeamIx, roofBeamZx) = getProp(selectedRoofBeam)

	if(roofBeamZx < ZBeamRoofReq):
		# print("The beam does not meet Zx requirement. Reselecting...")
		qualifiedZx 		= qualifiedIx[qualifiedIx['Zx'] > ZBeamRoofReq]			# narrow list further down to only sufficient Zx
		sortedWeight 		= qualifiedZx.sort_values(by=['W'])						# select lightest from list
		selectedRoofBeam 		= sortedWeight.iloc[:1]
		(roofBeamAg, roofBeambf, roofBeamtf, roofBeamIx, roofBeamZx) = getProp(selectedRoofBeam)

	(roofBeamMn, roofBeamMpr, roofBeamVpr, roofBeamVGrav) 	= calcStrength(selectedRoofBeam, VGravRoof)

	# PH location check
	phVGravRoof 		= wLoad[-1]*LBay/2
	phVBeamRoof 		= 2*roofBeamMpr/(0.8*LBay)									# 0.9LBay for plastic hinge length
	phLocationRoof 		= phVBeamRoof > phVGravRoof

	if not phLocationRoof:
		# print('Detected plastic hinge away from ends. Reselecting...')
		ZBeamPHReq 			= wLoad[-1]*LBay**2/(4*Fy*Ry*Cpr)
		qualifiedZx 		= qualifiedIx[qualifiedIx['Zx'] > ZBeamPHReq]				# narrow list further down to only sufficient Zx
		sortedWeight 		= qualifiedZx.sort_values(by=['W'])							# select lightest from list
		selectedRoofBeam 		= sortedWeight.iloc[:1]
		(roofBeamAg, roofBeambf, roofBeamtf, roofBeamIx, roofBeamZx) 	= getProp(selectedRoofBeam)
		(roofBeamMn, roofBeamMpr, roofBeamVpr, roofBeamVGrav) 			= calcStrength(selectedRoofBeam, VGravRoof)

	# roof beam shear check
	roofBeamAweb 		= roofBeamAg - 2*(roofBeamtf*roofBeambf)
	roofBeamVn 			= 0.9*roofBeamAweb*0.6*Fy

	roofBeamShearFail 	= roofBeamVn < roofBeamVpr

	while roofBeamShearFail:
		# print('Beam shear check failed. Reselecting...')
		roofAgReq 				= 2*roofBeamVpr/(0.9*0.6*Fy)						# Assume web is half of gross area

		if 'qualifiedZx' in dir():
			qualifiedAg			= qualifiedZx[qualifiedZx['A'] > roofAgReq]			# narrow Zx list down to sufficient Ag
		else:
			qualifiedAg 		= qualifiedIx[qualifiedIx['A'] > roofAgReq]			# if Zx check wasn't done previously, use Ix list

		sortedWeight 		= qualifiedAg.sort_values(by=['W'])						# select lightest from list
		selectedRoofBeam 		= sortedWeight.iloc[:1]

		# recheck beam shear
		(roofBeamAg, roofBeambf, roofBeamtf, roofBeamIx, roofBeamZx) 	= getProp(selectedRoofBeam)
		(roofBeamMn, roofBeamMpr, roofBeamVpr, roofBeamVGrav) 			= calcStrength(selectedRoofBeam, VGravRoof)
		
		roofBeamAweb 		= roofBeamAg - 2*(roofBeamtf*roofBeambf)
		roofBeamVn 			= 0.9*roofBeamAweb*0.6*Fy
		
		roofBeamShearFail 	= roofBeamVn < roofBeamVpr

	############################################################################
	#              Columns

	# SCWB design

	Pr 				= np.empty(nFloor)
	Pr[-1] 			= beamVpr + roofBeamVGrav

	for i in range(nFloor-2, -1, -1):
		Pr[i] = beamVGrav + beamVpr + Pr[i + 1]

	# guess: use columns that has similar Ix to beam
	qualifiedIx 	= sortedCols[sortedCols['Ix'] > IBeamReq]								# eliminate all shapes with insufficient Ix
	selectedCol 	= qualifiedIx.iloc[(qualifiedIx['Ix'] - IBeamReq).abs().argsort()[:1]]	# select the first few that qualifies Ix

	(colAg, colbf, coltf, colIx, colZx) = getProp(selectedCol)

	colMpr 			= colZx*(Fy - Pr/colAg)

	# find required Zx for SCWB to be true
	scwbZReq 		= np.max(beamMpr/(Fy - Pr[:-1]/colAg))

	# select column based on SCWB
	qualifiedIx 	= sortedCols[sortedCols['Ix'] > IColReq]			# eliminate all shapes with insufficient Ix
	qualifiedZx 	= qualifiedIx[qualifiedIx['Zx'] > scwbZReq]			# eliminate all shapes with insufficient Z for SCWB
	sortedWeight 	= qualifiedZx.sort_values(by=['W'])					# select lightest from list
	selectedCol 	= sortedWeight.iloc[:1]

	(colAg, colbf, coltf, colIx, colZx) = getProp(selectedCol)

	colMpr 			= colZx*(Fy - Pr/colAg)

	# check final SCWB
	ratio 			= np.empty(nFloor-1)
	for i in range(nFloor-2, -1, -1):
		ratio[i] = (colMpr[i+1] + colMpr[i])/(2*beamMpr)
		if (ratio[i] < 1.0):
			print('SCWB check failed at floor ' + (nFloor+1) + ".")

	# column shear
	colAweb 		= colAg - 2*(coltf*colbf)
	colVn 			= 0.9*colAweb*0.6*Fy
	colVpr 			= max(colMpr/hCol)

	colShearFail 	= colVn < colVpr

	while colShearFail:
		# print('Column shear check failed. Reselecting...')
		AgReq 		= 2*colVpr/(0.9*0.6*Fy)										# Assume web is half of gross area

		qualifiedAg			= qualifiedZx[qualifiedZx['A'] > AgReq]				# narrow Zx list down to sufficient Ag

		sortedWeight 		= qualifiedAg.sort_values(by=['W'])					# select lightest from list
		selectedCol 		= sortedWeight.iloc[:1]

		# recheck column shear
		(colAg, colbf, coltf, colIx, colZx) = getProp(selectedCol)
		
		colAweb 		= colAg - 2*(coltf*colbf)
		colVn 			= 0.9*colAweb*0.6*Fy

		colMpr 			= colZx*(Fy - Pr/colAg)
		colVpr 			= max(colMpr/hCol)
		
		colShearFail 	= colVn < colVpr

	return(mu1, mu2, mu3, R1, R2, R3, moatGap, selectedBeam, selectedRoofBeam, selectedCol)

# if ran as standalone, display designs
if __name__ == '__main__':
	(mu1, mu2, mu3, R1, R2, R3, moatGap, selectedBeam, selectedRoofBeam, selectedCol) = design()
	muList 	= [mu1, mu2, mu3]
	RList 	= [R1, R2, R3]
	print('Selected friction coeffs: ', muList)
	print('Selected curvature radii: ', RList)
	print('Selected beam:')
	print(selectedBeam)
	print('Selected roof beam:')
	print(selectedRoofBeam)
	print('Selected column:')
	print(selectedCol)
	print('Moat gap:')
	print(moatGap)
