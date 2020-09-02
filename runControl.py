############################################################################
#             	Run control

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	August 2020

# Description: 	Main script
#				Manages files and writes input files for each run
# 				Calls LHS -> design -> buildModel -> eqAnly -> postprocessing
# 				Writes results in final csv file

# Open issues: 	(1) 

############################################################################

# import OpenSees and libraries
# from openseespy.opensees import *
# import math

# system commands
import os, os.path
import glob
import shutil

############################################################################
#              File management
############################################################################

# remove existing results
# explanation here: https://stackoverflow.com/a/31989328
def remove_thing(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

def empty_directory(path):
    for i in glob.glob(os.path.join(path, '*')):
        remove_thing(i)

empty_directory('outputs')

############################################################################
#              Perform runs
############################################################################
import pandas as pd
import LHS
import postprocessing
import eqAnly as eq

# initialize dataframe as an empty object
resultsDf 			= None

# generate LHS input sets
numRuns 						= 1
inputVariables, inputValues 	= LHS.generateInputs(numRuns)

# get ground motion database list
gmPath 			= "X:/Documents/bezerkeley/research/fpsScripts/frameOps/opsPython/groundMotions/"
databaseFile 	= 'gmList.csv'
gmDatabase 		= pd.read_csv(gmPath+databaseFile, index_col=None, header=0)

# for each input sets, write input files
for index, row in enumerate(inputValues):

	print('The run index is ' + str(index) + '.')					# run counter

	empty_directory('outputs')										# clear run histories

	# write input files as csv columns
	bearingIndex 	= pd.DataFrame(inputVariables, columns=['variable'])		# relies on ordering from LHS.py
	bearingValue 	= pd.DataFrame(row, columns=['value'])

	bearingIndex 	= bearingIndex.join(bearingValue)
	bearingIndex.to_csv('./inputs/bearingInput.csv', index=False)

	# for each input file, run all GMs in the database
	for ind in gmDatabase.index:

		filename 				= str(gmDatabase['filename'][ind])					# ground motion name
		filename 				= filename.replace('.AT2', '')						# remove extension from file name
		defFactor 				= float(gmDatabase['scaleFactor'][ind])				# scale factor used
		defS1 					= float(gmDatabase['gmS1'][ind])					# scaled pSa at T = 1s
	 		
	 	# move on to next set if bad friction coeffs encountered (handled in superStructDesign)
		try:
			runStatus, scaleFactor 		= eq.runGM(filename, defFactor, defS1)				# perform analysis (superStructDesign and buildModel imported within)
		except ValueError:
			print('Bearing solver returned negative friction coefficients. Skipping...')
			continue
		except TypeError:
			print('Bearing solver returned complex friction coefficients. Skipping...')
			continue

		resultsHeader, thisRun 	= postprocessing.failurePostprocess(filename, scaleFactor, runStatus)		# add run results to holder df

		# if initial run, start the dataframe with headers from postprocessing.py
		if resultsDf is None:
			resultsDf 			= pd.DataFrame(columns=resultsHeader)
			
		# add results onto the dataframe
		resultsDf 				= pd.concat([thisRun,resultsDf], sort=False)

resultsDf['Pi1'] 		= resultsDf['mu1']/resultsDf['S1']
resultsDf['Pi2'] 		= resultsDf['Tm']**2/(386.4/resultsDf['R1'])

# move columns for readability
cols 		= list(resultsDf)
cols.insert(0, cols.pop(cols.index('Pi2')))
cols.insert(0, cols.pop(cols.index('Pi1')))

resultsDf 	= resultsDf.loc[:,cols]
resultsDf.to_csv('./sessionOut/sessionSummary.csv', index=False)