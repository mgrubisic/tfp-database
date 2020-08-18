############################################################################
#             	Postprocessing tool

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	August 2020

# Description: 	Script reads /outputs/ values and extract relevant maxima
#				Results are returned in a DataFrame row

# Open issues: 	(1) Header name inconsistencies to be automatically handled
#				(2) automatically generate variables based on output.csv files
#				(3) intake scale from elsewhere
# 				(4) avoid rerunning design script

############################################################################

import pandas as pd
import numpy as np
import math

# functions to standardize csv output
def getShape(shape):
	shapeName 		= shape.iloc[0]['AISC_Manual_Label']
	return(shapeName)

def getHeader():
	resultsHeader  	= ['S1', 'S1Ampli', 'Tm', 'zetaM', 'RI', 'mu1', 'mu2', 'mu3', 'beam', 'roofBeam', 'column', 'R1', 'R2', 'R3', 'moatAmplification', 'moatGap', 'GMFile', 'GMScale', 'maximumIsolDispl', 'driftMax1', 'driftMax2', 'driftMax3', 'drift1', 'drift2', 'drift3', 'impacted', 'uplifted', 'runFailed']
	return(resultsHeader)

def failurePostprocess(filename, defFactor, runStatus):

	# take input as the run 'ok' variable from eqAnly, passes that as one of the results csv columns

	# gather inputs
	bearingParams = pd.read_csv('./inputs/bearingInput.csv', index_col=None, header=0)
	S1 			= bearingParams.value[0]		# site spectrum value (either Sd1 or Sm1)
	S1Ampli 	= bearingParams.value[1]		# site amplification
	Tm 			= bearingParams.value[2]		# target period
	zetaM 		= bearingParams.value[3]		# target damping
	moatAmpli 	= bearingParams.value[6] 		# moat gap amplification

	buildingParams = pd.read_csv('./inputs/buildingInput.csv', index_col=None, header=0)
	RI 		= buildingParams.value[2]		# strength reduction factor R

	# redo scaling
	S1Actual 		= S1*S1Ampli
	S1Default 		= 1.017
	GMFactor 		= S1Actual/S1Default*defFactor

	import superStructDesign as sd
	(mu1, mu2, mu3, R1, R2, R3, moatGap, selectedBeam, selectedRoofBeam, selectedCol) = sd.design()

	beamName 		= getShape(selectedBeam)
	roofBeamName 	= getShape(selectedRoofBeam)
	colName 		= getShape(selectedCol)

	# gather outputs
	dispColumns = ['time', 'isol1', 'isol2', 'isol3', 'isol4', 'isolLC']

	isolDisp = pd.read_csv('./outputs/isolDisp.csv', sep=' ', header=None, names=dispColumns)
	isolVert = pd.read_csv('./outputs/isolVert.csv', sep=' ', header=None, names=dispColumns)
	isolRot  = pd.read_csv('./outputs/isolRot.csv', sep=' ', header=None, names=dispColumns)

	story1Disp = pd.read_csv('./outputs/story1Disp.csv', sep=' ', header=None, names=dispColumns)
	story2Disp = pd.read_csv('./outputs/story2Disp.csv', sep=' ', header=None, names=dispColumns)
	story3Disp = pd.read_csv('./outputs/story3Disp.csv', sep=' ', header=None, names=dispColumns)

	forceColumns = ['time', 'iAxial', 'iShearX', 'iShearY', 'iMomentX', 'iMomentY', 'iMomentZ', 'jAxial', 'jShearX', 'jShearY', 'jMomentX', 'jMomentY', 'jMomentZ']

	isol1Force = pd.read_csv('./outputs/isol1Force.csv', sep = ' ', header=None, names=forceColumns)
	isol2Force = pd.read_csv('./outputs/isol2Force.csv', sep = ' ', header=None, names=forceColumns)
	isol3Force = pd.read_csv('./outputs/isol3Force.csv', sep = ' ', header=None, names=forceColumns)
	isol4Force = pd.read_csv('./outputs/isol4Force.csv', sep = ' ', header=None, names=forceColumns)

	# maximum displacements across all isolators
	isol1Disp 		= abs(isolDisp['isol1'])
	isol2Disp 		= abs(isolDisp['isol2'])
	isol3Disp 		= abs(isolDisp['isol3'])
	isol4Disp 		= abs(isolDisp['isol4'])
	isolMaxDisp 	= np.maximum.reduce([isol1Disp, isol2Disp, isol3Disp, isol4Disp])

	# normalized shear in isolators
	force1Normalize = -isol1Force['iShearX']/isol1Force['iAxial']
	force2Normalize = -isol2Force['iShearX']/isol2Force['iAxial']
	force3Normalize = -isol3Force['iShearX']/isol3Force['iAxial']
	force4Normalize = -isol4Force['iShearX']/isol4Force['iAxial']

	# drift ratios recorded
	story1DriftOuter 	= (story1Disp['isol1'] - isolDisp['isol1'])/(13*12)
	story1DriftInner 	= (story1Disp['isol2'] - isolDisp['isol2'])/(13*12)

	story2DriftOuter 	= (story2Disp['isol1'] - story1Disp['isol1'])/(13*12)
	story2DriftInner 	= (story2Disp['isol2'] - story1Disp['isol2'])/(13*12)

	story3DriftOuter 	= (story3Disp['isol1'] - story2Disp['isol1'])/(13*12)
	story3DriftInner 	= (story3Disp['isol2'] - story2Disp['isol2'])/(13*12)

	# drift failure check
	driftLimit 			= 0.05

	driftMax1 			= max(np.maximum(story1DriftOuter, story1DriftInner))
	driftMax2 			= max(np.maximum(story2DriftOuter, story2DriftInner))
	driftMax3 			= max(np.maximum(story3DriftOuter, story3DriftInner))

	drift1				= 0
	drift2 				= 0
	drift3 				= 0

	if(any(abs(driftRatio) > driftLimit for driftRatio in story1DriftOuter) or any(abs(driftRatio) > driftLimit for driftRatio in story1DriftInner)):
		drift1 	= 1

	if(any(abs(driftRatio) > driftLimit for driftRatio in story2DriftOuter) or any(abs(driftRatio) > driftLimit for driftRatio in story2DriftInner)):
		drift2 	= 1

	if(any(abs(driftRatio) > driftLimit for driftRatio in story3DriftOuter) or any(abs(driftRatio) > driftLimit for driftRatio in story3DriftInner)):
		drift3 	= 1

	# impact check
	moatGap 			= float(moatGap)
	# moatGap 			= 20
	impacted 			= 0
	maxDisplacement 	= max(isolMaxDisp) 					# max recorded displacement over time

	if(any(displacement >= moatGap for displacement in isolMaxDisp)):
		impacted 		= 1

	# uplift check
	minFv 				= 5.0			# kips
	uplifted 			= 0

	# isolator axial forces
	isol1Axial 		= abs(isol1Force['iAxial'])
	isol2Axial 		= abs(isol2Force['iAxial'])
	isol3Axial 		= abs(isol3Force['iAxial'])
	isol4Axial 		= abs(isol4Force['iAxial'])

	isolMinAxial 	= np.minimum.reduce([isol1Axial, isol2Axial, isol3Axial, isol4Axial])

	if(any(axialForce <= minFv for axialForce in isolMinAxial)):
		uplifted 	= 1

	outHeader  		= getHeader()
	runRecord 		= pd.DataFrame([[S1, S1Ampli, Tm, zetaM, RI, mu1, mu2, mu3, beamName, roofBeamName, colName, R1, R2, R3, moatAmpli, moatGap, filename, GMFactor, maxDisplacement, driftMax1, driftMax2, driftMax3, drift1, drift2, drift3, impacted, uplifted, runStatus]], columns=outHeader)

	return(runRecord)

if __name__ == '__main__':
	thisRun 		= failurePostprocess(0)
	print(thisRun)