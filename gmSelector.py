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
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn', ignore SettingWithCopyWarning

def cleanGMs(gmDir, resultsCSV, actualS1):

	# remove all DT2 VT2 files
	folder 				= os.listdir(gmDir)

	for item in folder:
		if item.endswith('.VT2') or item.endswith('.DT2'):
			os.remove(os.path.join(gmDir,item))

	# load in sections of the sheet
	summary 			= pd.read_csv(gmDir+resultsCSV, skiprows=33, nrows=100)
	scaledSpectra 		= pd.read_csv(gmDir+resultsCSV, skiprows=144, nrows=111)
	unscaledSpectra 	= pd.read_csv(gmDir+resultsCSV, skiprows=258, nrows=111)

	# Keep Ss as 2.2815 (Berkeley)
	Ss 									= 2.2815
	Tshort								= actualS1/Ss
	targetSpectrum 						= scaledSpectra[['Period (sec)']]
	targetSpectrum['Target pSa (g)'] 	= np.where(targetSpectrum['Period (sec)'] < Tshort, Ss, actualS1/targetSpectrum['Period (sec)'])
	# selectionScaledSpectra				= scaledSpectra[selectionGMs]

	# calculate desired target spectrum average (0.2*T1, 3*T1)
	tLower 					= 0.6
	tUpper					= 4.5

	# geometric mean from Eads et al. (2015)
	targetRange 			= targetSpectrum[targetSpectrum['Period (sec)'].between(tLower, tUpper)]['Target pSa (g)']
	targetAverage 			= targetRange.prod()**(1/targetRange.size)

	# get the spectrum average for the unscaled GM spectra
	unscaledH1s 			= unscaledSpectra.filter(regex=("-1 pSa \(g\)$"))		# only concerned about H1 spectra
	unscaledSpectraRange 	= unscaledH1s[targetSpectrum['Period (sec)'].between(tLower, tUpper)]
	unscaledAverages 		= unscaledSpectraRange.prod()**(1/len(unscaledSpectraRange.index))

	# determine scale factor to get unscaled to target
	scaleFactorAverage 			= targetAverage/unscaledAverages
	scaleFactorAverage 			= scaleFactorAverage.reset_index()
	scaleFactorAverage.columns 	= ['fullRSN', 'avgSpectrumScaleFactor']

	# rename back to old convention and merge with previous dataframe
	scaleFactorAverage[' Record Sequence Number'] 		= scaleFactorAverage['fullRSN'].str.extract('(\d+)')
	scaleFactorAverage 			= scaleFactorAverage.astype({' Record Sequence Number': int})
	summary 					= pd.merge(summary, scaleFactorAverage, on=' Record Sequence Number').drop(columns=['fullRSN'])
	

	# grab only relevant columns
	summaryNames 		= [' Record Sequence Number', 'avgSpectrumScaleFactor', ' Earthquake Name', ' Lowest Useable Frequency (Hz)', ' Horizontal-1 Acc. Filename']
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
		matchingEqs['scaleDifference'] 		= abs(matchingEqs['avgSpectrumScaleFactor'] - 1.0)
		leastScaled 						= matchingEqs[matchingEqs['scaleDifference'] == min(matchingEqs['scaleDifference'])]

		if finalGM is None:
			GMHeaders 						= list(matchingEqs.columns)
			finalGM 						= pd.DataFrame(columns=GMHeaders)
		
		finalGM 							= pd.concat([leastScaled,finalGM], sort=False)
		finalGM[' Horizontal-1 Acc. Filename'] 	= finalGM[' Horizontal-1 Acc. Filename'].str.strip()

	# match new list with headers from spectra section
	# selectionGMs 			= [('RSN-' + str(rsn) + ' H1 pSa (g)') for rsn in finalGM[' Record Sequence Number']]
	selectionUnscaledGMs 	= [('RSN-' + str(rsn) + ' Horizontal-1 pSa (g)') for rsn in finalGM[' Record Sequence Number']]

	# find row where spectra is at T = 1s, return as column dataframe
	pSaOneSec 				= unscaledSpectra[selectionUnscaledGMs].loc[targetSpectrum['Period (sec)'] == 1].transpose().reset_index()
	pSaOneSec.columns 		= ['fullRSN', 'unscaledSa1']

	# rename back to old convention and merge with previous dataframe
	pSaOneSec[' Record Sequence Number'] 		= pSaOneSec['fullRSN'].str.extract('(\d+)')		# extract digits
	pSaOneSec 				= pSaOneSec.astype({' Record Sequence Number': int})
	finalGM 		    	= pd.merge(finalGM, pSaOneSec, on=' Record Sequence Number').drop(columns=['fullRSN', 'scaleDifference'])
	finalGM['scaledSa1'] 	= finalGM['avgSpectrumScaleFactor']*finalGM['unscaledSa1']

	finalGM.columns 			= ['RSN', 'scaleFactorSpecAvg', 'EQName', 'lowestFreq', 'filename', 'unscaledSa1', 'scaledSa1']

	return(finalGM, targetAverage)

def getST(gmDir, resultsCSV, GMFile, scaleFactor, Tquery):

	import re

	# load in sections of the sheet
	summary 			= pd.read_csv(gmDir+resultsCSV, skiprows=33, nrows=100)
	unscaledSpectra 	= pd.read_csv(gmDir+resultsCSV, skiprows=258, nrows=111)

	rsn 				= re.search('(\d+)', GMFile).group(1)
	gmUnscaledName		= 'RSN-' + str(rsn) + ' Horizontal-1 pSa (g)'
	gmSpectrum			= unscaledSpectra[['Period (sec)', gmUnscaledName]]
	gmSpectrum.columns	= ['Period', 'Sa']

	SaQueryUnscaled 	= np.interp(Tquery, gmSpectrum.Period, gmSpectrum.Sa)
	SaQuery 			= scaleFactor*SaQueryUnscaled
	return(SaQuery)



# Specify locations
if __name__ == '__main__':
	gmFolder 	= './groundMotions/PEERNGARecords_Unscaled/'
	PEERSummary = '_SearchResults.csv'
	gmDatabase 	= 'gmList.csv'
	testerS1 	= 1.15

	gmDf, specAvg 		= cleanGMs(gmFolder, PEERSummary, testerS1)
	gmDf.to_csv(gmFolder+gmDatabase, index=False)