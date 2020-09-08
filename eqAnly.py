############################################################################
#             	Earthquake analysis

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	May 2020

# Description: 	Script performs dynamic analysis on OpenSeesPy model

# Open issues: 	(1) Algorithm robustness to be checked

############################################################################


# import OpenSees and libraries
from openseespy.opensees import *
import math

############################################################################
#              Build model
############################################################################

import buildModel as bm
import pandas as pd

def runGM(gmFilename, gmDefScale, gmDefS1):

	# build model
	bm.build()

	# # clear analysis elements and return structure to unloaded state
	# # for some reason, building once and resetting analysis is slower than just wiping and rebuilding the whole thing
	# wipeAnalysis()
	# reset()

	(w0, w1, w2, w3, pLc0, pLc1, pLc2, pLc3) = bm.giveLoads()

	############################################################################
	#              Clear previous runs
	############################################################################

	gravSeriesTag 	= 1
	gravPatternTag 	= 1

	eqSeriesTag 	= 100
	eqPatternTag 	= 400

	# remove('timeSeries', gravSeriesTag)
	# remove('loadPattern', gravPatternTag)
	# remove('timeSeries', eqSeriesTag)
	# remove('loadPattern', eqPatternTag)


	############################################################################
	#              Loading and analysis
	############################################################################

	from ReadRecord import ReadRecord

	# ------------------------------
	# Loading: gravity
	# ------------------------------

	# create TimeSeries
	timeSeries("Linear", gravSeriesTag)

	# create plain load pattern
	# command: 	pattern('Plain', tag, timeSeriesTag)
	# command:  eleLoad('-ele', *eleTags, '-range', eleTag1, eleTag2, '-type', '-beamUniform', Wy, Wz=0.0, Wx=0.0, '-beamPoint', Py, Pz=0.0, xL, Px=0.0, '-beamThermal', *tempPts)
	# command:  load(nodeTag, *loadValues)
	pattern('Plain', gravPatternTag, gravSeriesTag)
	eleLoad('-ele', 38, 39, 40, '-type', '-beamUniform', -w0, 0)
	eleLoad('-ele', 13, 14, 15, '-type', '-beamUniform', -w1, 0)
	eleLoad('-ele', 16, 17, 18, '-type', '-beamUniform', -w2, 0)
	eleLoad('-ele', 19, 20, 21, '-type', '-beamUniform', -w3, 0)

	load(22, 0, 0, -pLc0, 0, 0, 0)
	load(24, 0, 0, -pLc1, 0, 0, 0)
	load(27, 0, 0, -pLc2, 0, 0, 0)
	load(30, 0, 0, -pLc3, 0, 0, 0)


	# ------------------------------
	# Start of analysis generation: gravity
	# ------------------------------

	nStepGravity 	= 10					# apply gravity in 10 steps
	tol 			= 1e-5					# convergence tolerance for test
	dGravity 		= 1/nStepGravity		# first load increment

	system("BandGeneral")					# create system of equation (SOE)
	test("NormDispIncr", tol, 15)			# determine if convergence has been achieved at the end of an iteration step
	numberer("RCM")							# create DOF numberer
	constraints("Plain")					# create constraint handler
	integrator("LoadControl", dGravity) 	# determine the next time step for an analysis, create integration scheme (steps of 1/10)
	algorithm("Newton")						# create solution algorithm
	analysis("Static")						# create analysis object
	analyze(nStepGravity)					# perform the analysis in N steps

	print("Gravity analysis complete!")

	loadConst('-time', 0.0)


	############################################################################
	#                       Eigenvalue Analysis
	############################################################################
	nEigenI 	= 1;					# mode i = 1
	nEigenJ 	= 3;					# mode j = 3
	lambdaN 	= eigen(nEigenJ);		# eigenvalue analysis for nEigenJ modes
	lambda1 	= lambdaN[0];			# eigenvalue mode i = 1
	lambda2		= lambdaN[1];			# eigenvalue mode j = 2
	omega1 		= math.sqrt(lambda1)	# w1 (1st mode circular frequency)
	omega2 		= math.sqrt(lambda2)	# w2 (2nd mode circular frequency)
	T1 			= 2*math.pi/omega1 		# 1st mode period of the structure
	T2 			= 2*math.pi/omega2 		# 2nd mode period of the structure				
	print("T1 = ", T1, " s")			# display the first mode period in the command window
	print("T2 = ", T2, " s")			# display the second mode period in the command window

	# record()
	# plot_modeshape(1)
	# plot_modeshape(2)

	# Rayleigh damping to the superstructure only
	regTag 		= 80
	zetaTarget 	= 0.05
	bm.provideSuperDamping(regTag, omega1, zetaTarget)

	############################################################################
	#              Recorders
	############################################################################

	dataDir 		= './outputs/'			# output folder
	#file mkdir $dataDir; # create output folder

	# record floor displacements
	printModel('-file', dataDir+'model.out')
	recorder('Node', '-file', dataDir+'isolDisp.csv', '-time', '-closeOnWrite', '-node', 1, 2, 3, 4, 22, '-dof', 1, 'disp')
	recorder('Node', '-file', dataDir+'isolVert.csv', '-time', '-closeOnWrite', '-node', 1, 2, 3, 4, 22, '-dof', 3, 'disp')
	recorder('Node', '-file', dataDir+'isolRot.csv', '-time', '-closeOnWrite', '-node', 1, 2, 3, 4, 22, '-dof', 5, 'disp')

	recorder('Node', '-file', dataDir+'story1Disp.csv', '-time', '-closeOnWrite', '-node', 5, 6, 7, 8, 24, '-dof', 1, 'disp')
	recorder('Node', '-file', dataDir+'story2Disp.csv', '-time', '-closeOnWrite', '-node', 9, 10, 11, 12, 27, '-dof', 1, 'disp')
	recorder('Node', '-file', dataDir+'story3Disp.csv', '-time', '-closeOnWrite', '-node', 13, 14, 15, 16, 30, '-dof', 1, 'disp')

	recorder('Element', '-file', dataDir+'isol1Force.csv', '-time', '-closeOnWrite', '-ele', 22, 'localForce')
	recorder('Element', '-file', dataDir+'isol2Force.csv', '-time', '-closeOnWrite', '-ele', 23, 'localForce')
	recorder('Element', '-file', dataDir+'isol3Force.csv', '-time', '-closeOnWrite', '-ele', 24, 'localForce')
	recorder('Element', '-file', dataDir+'isol4Force.csv', '-time', '-closeOnWrite', '-ele', 25, 'localForce')
	recorder('Element', '-file', dataDir+'isolLCForce.csv', '-time', '-closeOnWrite', '-ele', 26, 'localForce')

	recorder('Element', '-file', dataDir+'colForce1.csv', '-time', '-closeOnWrite', '-ele', 1, 'localForce')
	recorder('Element', '-file', dataDir+'colForce2.csv', '-time', '-closeOnWrite', '-ele', 4, 'localForce')
	recorder('Element', '-file', dataDir+'colForce3.csv', '-time', '-closeOnWrite', '-ele', 7, 'localForce')
	recorder('Element', '-file', dataDir+'colForce4.csv', '-time', '-closeOnWrite', '-ele', 10, 'localForce')

	recorder('Element', '-file', dataDir+'colForce5.csv', '-time', '-closeOnWrite', '-ele', 2, 'localForce')
	recorder('Element', '-file', dataDir+'colForce6.csv', '-time', '-closeOnWrite', '-ele', 5, 'localForce')
	recorder('Element', '-file', dataDir+'colForce7.csv', '-time', '-closeOnWrite', '-ele', 8, 'localForce')
	recorder('Element', '-file', dataDir+'colForce8.csv', '-time', '-closeOnWrite', '-ele', 11, 'localForce')

	recorder('Element', '-file', dataDir+'beamForce1.csv', '-time', '-closeOnWrite', '-ele', 13, 'localForce')
	recorder('Element', '-file', dataDir+'beamForce2.csv', '-time', '-closeOnWrite', '-ele', 14, 'localForce')
	recorder('Element', '-file', dataDir+'beamForce3.csv', '-time', '-closeOnWrite', '-ele', 15, 'localForce')

	recorder('Element', '-file', dataDir+'beamForce4.csv', '-time', '-closeOnWrite', '-ele', 16, 'localForce')
	recorder('Element', '-file', dataDir+'beamForce5.csv', '-time', '-closeOnWrite', '-ele', 17, 'localForce')
	recorder('Element', '-file', dataDir+'beamForce6.csv', '-time', '-closeOnWrite', '-ele', 18, 'localForce')

	recorder('Element', '-file', dataDir+'diaphragmForce1.csv', '-time', '-closeOnWrite', '-ele', 38, 'localForce')
	recorder('Element', '-file', dataDir+'diaphragmForce2.csv', '-time', '-closeOnWrite', '-ele', 39, 'localForce')
	recorder('Element', '-file', dataDir+'diaphragmForce3.csv', '-time', '-closeOnWrite', '-ele', 40, 'localForce')

	############################################################################
	#                       Dynamic analysis
	############################################################################

	wipeAnalysis()

	# Uniform Earthquake ground motion (uniform acceleration input at all support nodes)

	bearingParams 	= pd.read_csv('./inputs/bearingInput.csv', index_col=None, header=0)

	# param is dictionary of all inputs. call with param['whatYouWant']
	param 			= dict(zip(bearingParams.variable, bearingParams.value))

	actualS1 		= param['S1']*param['S1Ampli']


	GMDir 			= "./groundMotions/PEERNGARecords_Unscaled/"
	GMDirection 	= 1								# ground-motion direction
	GMFile 			= gmFilename 					# ground motion file name passed in

	# Search result is scaled to Sm1 = gmDefS1, so scale appropriately
	GMFactor 		= actualS1/gmDefS1*gmDefScale

	print('Current ground motion: ', gmFilename)

	# Set up Analysis Parameters ---------------------------------------------
	# CONSTRAINTS handler -- Determines how the constraint equations are enforced in the analysis (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/617.htm)
	#          Plain Constraints -- Removes constrained degrees of freedom from the system of equations 
	#          Lagrange Multipliers -- Uses the method of Lagrange multipliers to enforce constraints 
	#          Penalty Method -- Uses penalty numbers to enforce constraints 
	#          Transformation Method -- Performs a condensation of constrained degrees of freedom

	constraints('Plain')

	# DOF NUMBERER (number the degrees of freedom in the domain): (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/366.htm)
	#   determines the mapping between equation numbers and degrees-of-freedom
	#          Plain -- Uses the numbering provided by the user 
	#          RCM -- Renumbers the DOF to minimize the matrix band-width using the Reverse Cuthill-McKee algorithm 

	numberer('RCM')

	# SYSTEM (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/371.htm)
	#   Linear Equation Solvers (how to store and solve the system of equations in the analysis)
	#   -- provide the solution of the linear system of equations Ku = P. Each solver is tailored to a specific matrix topology. 
	#          ProfileSPD -- Direct profile solver for symmetric positive definite matrices 
	#          BandGeneral -- Direct solver for banded unsymmetric matrices 
	#          BandSPD -- Direct solver for banded symmetric positive definite matrices 
	#          SparseGeneral -- Direct solver for unsymmetric sparse matrices (-piv option)
	#          SparseSPD -- Direct solver for symmetric sparse matrices 
	#          UmfPack -- Direct UmfPack solver for unsymmetric matrices 

	system('BandGeneral')

	# TEST: # convergence test to 
	# Convergence TEST (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/360.htm)
	#   -- Accept the current state of the domain as being on the converged solution path 
	#   -- determine if convergence has been achieved at the end of an iteration step
	#          NormUnbalance -- Specifies a tolerance on the norm of the unbalanced load at the current iteration 
	#          NormDispIncr -- Specifies a tolerance on the norm of the displacement increments at the current iteration 
	#          EnergyIncr-- Specifies a tolerance on the inner product of the unbalanced load and displacement increments at the current iteration 
	#          RelativeNormUnbalance --
	#          RelativeNormDispIncr --
	#          RelativeEnergyIncr --
	  	
	tolDynamic 			= 1e-3		# Convergence Test: tolerance
	maxIterDynamic		= 50		# Convergence Test: maximum number of iterations that will be performed before "failure to converge" is returned
	printFlagDynamic 	= 0			# Convergence Test: flag used to print information on convergence (optional)        # 1: print information on each step; 
	testTypeDynamic		= 'NormDispIncr'
	test(testTypeDynamic, tolDynamic, maxIterDynamic, printFlagDynamic)

	# Solution ALGORITHM: -- Iterate from the last time step to the current (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/682.htm)
	#          Linear -- Uses the solution at the first iteration and continues 
	#          Newton -- Uses the tangent at the current iteration to iterate to convergence 
	#          ModifiedNewton -- Uses the tangent at the first iteration to iterate to convergence 
	#          NewtonLineSearch -- 
	#          KrylovNewton -- 
	#          BFGS -- 
	#          Broyden -- 
	algorithmTypeDynamic	= 'Broyden'
	algorithm(algorithmTypeDynamic, 8)

	# Static INTEGRATOR: -- determine the next time step for an analysis  (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/689.htm)
	#          LoadControl -- Specifies the incremental load factor to be applied to the loads in the domain 
	#          DisplacementControl -- Specifies the incremental displacement at a specified DOF in the domain 
	#          Minimum Unbalanced Displacement Norm -- Specifies the incremental load factor such that the residual displacement norm in minimized 
	#          Arc Length -- Specifies the incremental arc-length of the load-displacement path 
	# Transient INTEGRATOR: -- determine the next time step for an analysis including inertial effects 
	#          Newmark -- The two parameter time-stepping method developed by Newmark 
	#          HHT -- The three parameter Hilbert-Hughes-Taylor time-stepping method 
	#          Central Difference -- Approximates velocity and acceleration by centered finite differences of displacement 

	newmarkGamma 	= 0.5			# Newmark-integrator gamma parameter (also HHT)
	newmarkBeta		= 0.25			# Newmark-integrator beta parameter
	integrator('Newmark', newmarkGamma, newmarkBeta)

	# ANALYSIS  -- defines what type of analysis is to be performed (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/324.htm)
	#          Static Analysis -- solves the KU=R problem, without the mass or damping matrices. 
	#          Transient Analysis -- solves the time-dependent analysis. The time step in this type of analysis is constant. The time step in the output is also constant. 
	#          variableTransient Analysis -- performs the same analysis type as the Transient Analysis object. The time step, however, is variable. This method is used when 
	#                 there are convergence problems with the Transient Analysis object at a peak or when the time step is too small. The time step in the output is also variable.

	analysis('Transient')

	#  ---------------------------------    perform Dynamic Ground-Motion Analysis
	# the following commands are unique to the Uniform Earthquake excitation

	# Uniform EXCITATION: acceleration input
	inFile 			= GMDir + GMFile + '.AT2'
	outFile 		= GMDir + GMFile + '.g3'		# set variable holding new filename (PEER files have .at2/dt2 extension)

	dt, nPts		= ReadRecord(inFile, outFile)	# call procedure to convert the ground-motion file
	g 				= 386.4
	GMfatt			= g*GMFactor					# data in input file is in g Unifts -- ACCELERATION TH

	timeSeries('Path', eqSeriesTag, '-dt', dt, '-filePath', outFile, '-factor', GMfatt)		# time series information
	pattern('UniformExcitation', eqPatternTag, GMDirection, '-accel', eqSeriesTag)			# create uniform excitation

	# set up ground-motion-analysis parameters
	sec 			= 1.0
	DtAnalysis		= 0.005*sec						# time-step Dt for lateral analysis                      	
	GMTime			= dt*nPts + 10                  # total time of ground motion + 10 sec of free vibration
	# TmaxAnalysis	= GMTime*sec					# maximum duration of ground-motion analysis
	TmaxAnalysis 	= 60.0*sec

	Nsteps			= math.floor(TmaxAnalysis/DtAnalysis)
	ok				= analyze(Nsteps, DtAnalysis)	# actually perform analysis; returns ok=0 if analysis was successful

	# analysis was not successful.
	if ok != 0:
		# --------------------------------------------------------------------------------------------------
		# change some analysis parameters to achieve convergence
		# performance is slower inside this loop
		#    Time-controlled analysis
		ok				= 0
		controlTime		= getTime()
		while (controlTime < TmaxAnalysis) and (ok == 0):
			controlTime	= getTime()
			ok			= analyze(1, DtAnalysis)
			print("Convergence issues at time: ", controlTime)
			if ok != 0:
				print("Trying Newton with Initial Tangent...")
				test('NormDispIncr', tolDynamic, 100, 0)
				algorithm('Newton', '-initial')
				ok = analyze(1, DtAnalysis)
				if ok == 0:
					print("That worked. Back to Broyden")
				test(testTypeDynamic, tolDynamic, maxIterDynamic, 0)
				algorithm(algorithmTypeDynamic, 8)

			if ok != 0:
				print('Trying Newton ...')
				algorithm('Newton')
				ok = analyze(1, DtAnalysis)
				if ok == 0:
					print("That worked. Back to Broyden")
				algorithm(algorithmTypeDynamic, 8)

			if ok != 0:
				print('Trying NewtonWithLineSearch ...')
				algorithm('NewtonLineSearch', 0.8)
				ok = analyze(1, DtAnalysis)
				if ok == 0:
					print("That worked. Back to Broyden")
				algorithm(algorithmTypeDynamic, 8)

			if ok != 0:
				print('Trying Newton Raphson ...')
				algorithm('RaphsonNewton')
				ok = analyze(1, DtAnalysis)
				if ok == 0:
					print("That worked. Back to Broyden")
				algorithm(algorithmTypeDynamic, 8)

			if ok != 0:
				print('Trying Krylov Newton ...')
				algorithm('KrylovNewton ')
				ok = analyze(1, DtAnalysis)
				if ok == 0:
					print("That worked. Back to Broyden")
				algorithm(algorithmTypeDynamic, 8)

			if ok != 0:
				print('Trying BFGS ...')
				algorithm('BFGS ')
				ok = analyze(1, DtAnalysis)
				if ok == 0:
					print("That worked. Back to Broyden")
				algorithm(algorithmTypeDynamic, 8)

	print('Ground motion done. End time:', getTime())

	wipe()

	return(ok, GMFactor)