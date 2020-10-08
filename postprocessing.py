############################################################################
#             	Postprocessing tool

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	August 2020

# Description: 	Script reads /outputs/ values and extract relevant maxima
#				Results are returned in a DataFrame row
#				Results are created and called using dictionaries

# Open issues: 	(1) avoid rerunning design script
#				(2) GMDir currently hardcoded here
# 				(3) Pi groups currently deferred to R

############################################################################

import pandas as pd
import numpy as np
import math
import gmSelector

# functions to standardize csv output
def getShape(shape):
	shapeName 		= shape.iloc[0]['AISC_Manual_Label']
	return(shapeName)

# main function
def failurePostprocess(filename, scaleFactor, spectrumAverage, runStatus, Tfb):

	# take input as the run 'ok' variable from eqAnly, passes that as one of the results csv columns

	# gather inputs
	bearingParams 			= pd.read_csv('./inputs/bearingInput.csv', index_col=None, header=0)

	# param is dictionary of all inputs. call with param['whatYouWant']
	param 					= dict(zip(bearingParams.variable, bearingParams.value))

	# create new dictionary for non-inputs. Put all tabulated results here.
	afterRun 				= dict()

	# scaling and filename are read
	afterRun['GMFile'] 		= filename
	afterRun['GMScale'] 	= scaleFactor
	

	# get selections and append to non-input dictionary
	import superStructDesign as sd
	(mu1, mu2, mu3, R1, R2, R3, moatGap, selectedBeam, selectedRoofBeam, selectedCol) = sd.design()

	fromDesign 		= {
		'mu1'		: mu1,
		'mu2'		: mu2,
		'mu3'		: mu3,
		'R1'		: R1,
		'R2'		: R2,
		'R3'		: R3,
		'beam'		: getShape(selectedBeam),
		'roofBeam'	: getShape(selectedRoofBeam),
		'col'		: getShape(selectedCol),
		'moatGap'	: float(moatGap),
	}

	afterRun.update(fromDesign)

	gmDir 					= './groundMotions/PEERNGARecords_Unscaled/'
	resultsCSV				= 'combinedSearch.csv'

	# calculate system periods
	afterRun['Tfb'] 		= Tfb
	afterRun['T1']			= 2*math.pi/(math.sqrt(afterRun['mu2']*386.4/(2*afterRun['R1']*(afterRun['mu2'] - afterRun['mu1']))))
	afterRun['T2'] 			= 2*math.pi/(math.sqrt(386.4/(2*afterRun['R2'])))

	# spectral accels of target spectrum
	afterRun['ST1']			= param['S1']/afterRun['T1']
	afterRun['ST2'] 		= param['S1']/afterRun['T2']

	# spectral accels of GM
	afterRun['GMSavg'] 		= spectrumAverage
	afterRun['GMS1'] 		= gmSelector.getST(gmDir, resultsCSV, filename, scaleFactor, 1.0, 32, 133, 290, 111)
	afterRun['GMST1']		= gmSelector.getST(gmDir, resultsCSV, filename, scaleFactor, afterRun['T1'], 32, 133, 290, 111)
	afterRun['GMST2']		= gmSelector.getST(gmDir, resultsCSV, filename, scaleFactor, afterRun['T2'], 32, 133, 290, 111)
	afterRun['GMSTm'] 		= gmSelector.getST(gmDir, resultsCSV, filename, scaleFactor, param['Tm'], 32, 133, 290, 111)

	# # calculate nondimensionalized parameters
	# afterRun['Pi1'] 		= afterRun['mu1']/param['GMS1']
	# afterRun['Pi2'] 		= param['Tm']**2/(386.4/afterRun['R1'])
	# afterRun['Pi3']			= afterRun['T2']/afterRun['T1']
	# afterRun['Pi4']			= afterRun['mu2']/afterRun['GMST2']

	# gather outputs
	dispColumns = ['time', 'isol1', 'isol2', 'isol3', 'isol4', 'isolLC']

	isolDisp = pd.read_csv('./outputs/isolDisp.csv', sep=' ', header=None, names=dispColumns)

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

	afterRun['maxDisplacement'] 	= max(isolMaxDisp) 					# max recorded displacement over time

	# drift ratios recorded
	ft 				= 12
	story1DriftOuter 	= (story1Disp['isol1'] - isolDisp['isol1'])/(13*ft)
	story1DriftInner 	= (story1Disp['isol2'] - isolDisp['isol2'])/(13*ft)

	story2DriftOuter 	= (story2Disp['isol1'] - story1Disp['isol1'])/(13*ft)
	story2DriftInner 	= (story2Disp['isol2'] - story1Disp['isol2'])/(13*ft)

	story3DriftOuter 	= (story3Disp['isol1'] - story2Disp['isol1'])/(13*ft)
	story3DriftInner 	= (story3Disp['isol2'] - story2Disp['isol2'])/(13*ft)

	# drift failure check
	# collapse limit state
	collapseDriftLimit 			= 0.05

	afterRun['driftMax1']	= max(np.maximum(abs(story1DriftOuter), abs(story1DriftInner)))
	afterRun['driftMax2']	= max(np.maximum(abs(story2DriftOuter), abs(story2DriftInner)))
	afterRun['driftMax3']	= max(np.maximum(abs(story3DriftOuter), abs(story3DriftInner)))

	afterRun['collapseDrift1']	= 0
	afterRun['collapseDrift2'] 	= 0
	afterRun['collapseDrift3'] 	= 0

	if(any(abs(driftRatio) > collapseDriftLimit for driftRatio in story1DriftOuter) or any(abs(driftRatio) > collapseDriftLimit for driftRatio in story1DriftInner)):
		afterRun['collapseDrift1'] 	= 1

	if(any(abs(driftRatio) > collapseDriftLimit for driftRatio in story2DriftOuter) or any(abs(driftRatio) > collapseDriftLimit for driftRatio in story2DriftInner)):
		afterRun['collapseDrift2'] 	= 1

	if(any(abs(driftRatio) > collapseDriftLimit for driftRatio in story3DriftOuter) or any(abs(driftRatio) > collapseDriftLimit for driftRatio in story3DriftInner)):
		afterRun['collapseDrift3'] 	= 1


	serviceDriftLimit 			= 0.015 	# need to find ASCE 41 basis

	afterRun['serviceDrift1']	= 0
	afterRun['serviceDrift2'] 	= 0
	afterRun['serviceDrift3'] 	= 0

	if(any(abs(driftRatio) > serviceDriftLimit for driftRatio in story1DriftOuter) or any(abs(driftRatio) > serviceDriftLimit for driftRatio in story1DriftInner)):
		afterRun['serviceDrift1'] 	= 1

	if(any(abs(driftRatio) > serviceDriftLimit for driftRatio in story2DriftOuter) or any(abs(driftRatio) > serviceDriftLimit for driftRatio in story2DriftInner)):
		afterRun['serviceDrift2'] 	= 1

	if(any(abs(driftRatio) > serviceDriftLimit for driftRatio in story3DriftOuter) or any(abs(driftRatio) > serviceDriftLimit for driftRatio in story3DriftInner)):
		afterRun['serviceDrift3'] 	= 1

	# impact check
	moatGap 				= float(moatGap)
	afterRun['impacted'] 	= 0

	if(any(displacement >= moatGap for displacement in isolMaxDisp)):
		afterRun['impacted'] 	= 1

	# uplift check
	minFv 					= 5.0			# kips
	afterRun['uplifted'] 	= 0

	# isolator axial forces
	isol1Axial 		= abs(isol1Force['iAxial'])
	isol2Axial 		= abs(isol2Force['iAxial'])
	isol3Axial 		= abs(isol3Force['iAxial'])
	isol4Axial 		= abs(isol4Force['iAxial'])

	isolMinAxial 	= np.minimum.reduce([isol1Axial, isol2Axial, isol3Axial, isol4Axial])

	if(any(axialForce <= minFv for axialForce in isolMinAxial)):
		afterRun['uplifted'] 	= 1

	# use run status passed in
	afterRun['runFailed'] 	= runStatus

	# merge input and output dictionaries, then output as dataframe
	runDict 		= {**param, **afterRun}
	runRecord 		= pd.DataFrame.from_dict(runDict, 'index').transpose()			# 'index' puts keys as index column. transpose so results are in a row.
	runHeader 		= list(runRecord.columns)										# pass column names to runControl
	
	return(runHeader, runRecord)

if __name__ == '__main__':
	thisHeader, thisRun 		= failurePostprocess('RSN3964_TOTTORI_TTR007NS', 3.0, 0.6, 1.21, 0)
	print(thisRun)