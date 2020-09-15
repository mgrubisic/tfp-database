############################################################################
#             	Ground motion selector

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	September 2020

# Description: 	Script creates list of viable ground motions and scales from PEER search

# Open issues: 	(1) Requires for PEER query to include Minimize MSE Scaling

############################################################################
import pandas as pd
import os

pd.options.mode.chained_assignment = None  # default='warn', ignore SettingWithCopyWarning

def cleanGMs(gmDir, resultsCSV):

	# remove all DT2 VT2 files
	folder 				= os.listdir(gmDir)

	for item in folder:
		if item.endswith('.VT2') or item.endswith('.DT2'):
			os.remove(os.path.join(gmDir,item))

	# load in sections of the sheet
	summary 			= pd.read_csv(gmDir+resultsCSV, skiprows=33, nrows=100)
	scaledSpectra 		= pd.read_csv(gmDir+resultsCSV, skiprows=144, nrows=111)
	unscaledSpectra 	= pd.read_csv(gmDir+resultsCSV, skiprows=258, nrows=111)

	# grab only relevant columns
	summaryNames 		= [' Record Sequence Number', ' Scale Factor', ' Earthquake Name', ' Lowest Useable Frequency (Hz)', ' Horizontal-1 Acc. Filename']
	summarySmol 		= summary[summaryNames]

	# Filter by lowest usable frequency
	TMax 				= 4.5
	freqMin 			= 1/TMax
	eligFreq 			= summarySmol[summarySmol[' Lowest Useable Frequency (Hz)'] < freqMin]

	# List unique earthquakes
	uniqEqs 			= pd.unique(eligFreq[' Earthquake Name'])
	finalGM 			= None

	# Select earthquakes that are least severely scaled
	for earthquake in uniqEqs:
		matchingEqs 						= eligFreq[eligFreq[' Earthquake Name'] == earthquake]
		matchingEqs['scaleDifference'] 		= abs(matchingEqs[' Scale Factor'] - 1.0)
		leastScaled 						= matchingEqs[matchingEqs['scaleDifference'] == min(matchingEqs['scaleDifference'])]

		if finalGM is None:
			GMHeaders 						= list(matchingEqs.columns)
			finalGM 						= pd.DataFrame(columns=GMHeaders)
		
		finalGM 							= pd.concat([leastScaled,finalGM], sort=False)
		finalGM[' Horizontal-1 Acc. Filename'] 	= finalGM[' Horizontal-1 Acc. Filename'].str.strip()

	# match new list with headers from spectra section
	selectionGMs 			= [('RSN-' + str(rsn) + ' H1 pSa (g)') for rsn in finalGM[' Record Sequence Number']]
	selectionUnscaledGMs 	= [('RSN-' + str(rsn) + ' Horizontal-1 pSa (g)') for rsn in finalGM[' Record Sequence Number']]

	targetSpectrum 			= scaledSpectra[['Period (sec)', 'Target pSa (g)']]
	selectionScaledSpectra	= scaledSpectra[selectionGMs]

	# find row where spectra is at T = 1s, return as column dataframe
	pSaOneSec 				= selectionScaledSpectra.loc[targetSpectrum['Period (sec)'] == 1].transpose().reset_index()
	pSaOneSec.columns 		= ['fullRSN', 'scaledSa1']

	# rename back to old convention and merge with previous dataframe
	pSaOneSec[' Record Sequence Number'] 		= pSaOneSec['fullRSN'].str.extract('(\d+)')		# extract digits
	pSaOneSec 				= pSaOneSec.astype({' Record Sequence Number': int})
	finalGM 		    	= pd.merge(finalGM, pSaOneSec, on=' Record Sequence Number').drop(columns=['fullRSN', 'scaleDifference'])

	# calculate desired target spectrum average (0.2*T1, 2.0*T1) 
	tLower 					= 0.6
	tUpper					= 4.5

	# geometric mean from Eads et al. (2015)
	targetRange 			= targetSpectrum[targetSpectrum['Period (sec)'].between(tLower, tUpper)]['Target pSa (g)']
	targetAverage 			= targetRange.prod()**(1/targetRange.size)

	# get the spectrum average for the unscaled GM spectra
	unscaledSpectraRange 	= unscaledSpectra[selectionUnscaledGMs][targetSpectrum['Period (sec)'].between(tLower, tUpper)]
	unscaledAverages 		= unscaledSpectraRange.prod()**(1/len(unscaledSpectraRange.index))

	# determine scale factor to get unscaled to target
	scaleFactorAverage 			= targetAverage/unscaledAverages
	scaleFactorAverage 			= scaleFactorAverage.reset_index()
	scaleFactorAverage.columns 	= ['fullRSN', 'avgSpectrumScaleFactor']

	# rename back to old convention and merge with previous dataframe
	scaleFactorAverage[' Record Sequence Number'] 		= scaleFactorAverage['fullRSN'].str.extract('(\d+)')
	scaleFactorAverage 			= scaleFactorAverage.astype({' Record Sequence Number': int})
	finalGM 		    		= pd.merge(finalGM, scaleFactorAverage, on=' Record Sequence Number').drop(columns=['fullRSN'])
	finalGM.columns 			= ['RSN', 'scaleFactorS1', 'EQName', 'lowestFreq', 'filename', 'scaledSa1', 'scaleFactorSpecAvg']

	return(finalGM)

# Specify locations
if __name__ == '__main__':
	gmFolder 	= './groundMotions/PEERNGARecords_Unscaled/'
	PEERSummary = '_SearchResults.csv'
	gmDatabase 	= 'gmList.csv'

	gmDf 		= cleanGMs(gmFolder, PEERSummary)
	gmDf.to_csv(gmFolder+gmDatabase, index=False)