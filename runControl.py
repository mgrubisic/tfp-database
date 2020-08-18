############################################################################
#             	Run control

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	August 2020

# Description: 	Main script
#				Manages files and writes input files for each run
# 				Calls LHS -> design -> buildModel -> eqAnly -> postprocessing
# 				Writes results in final csv file

# Open issues: 	(1) Fix workflow
# 					-avoid repeating modeling building for each GM run
# 				(2) Cleanly pass inputs around rather than rereading
#				(3) Inputs currently hardcoded

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

# create dataframe to hold results
resultsHeader  	= postprocessing.getHeader()
resultsDf 		= pd.DataFrame(columns=resultsHeader)

# generate LHS input sets
numRuns 		= 1
inputSet 		= LHS.generateInputs(numRuns)

# get ground motion database list
gmPath 			= "X:/Documents/bezerkeley/research/fpsScripts/frameOps/opsPython/groundMotions/"
databaseFile 	= 'gmList.csv'
gmDatabase 		= pd.read_csv(gmPath+databaseFile, index_col=None, header=0)

# for each input sets, write input files
for index, row in enumerate(inputSet):

	print('The run index is ' + str(index) + '.')

	empty_directory('outputs')										# clear run histories

	# write input files as csv columns
	bearingIndex 	= pd.DataFrame(['Sm1', 'Sm1Ampli', 'T', 'zeta', 'mu1', 'R1', 'moatAmpli'], columns=['variable'])
	bearingValue 	= pd.DataFrame([row[0], row[1], row[2], row[3], row[4], row[5], row[6]], columns=['value'])

	bearingIndex 	= bearingIndex.join(bearingValue)
	bearingIndex.to_csv('./inputs/bearingInput.csv', index=False)
	
	buildingIndex 	= pd.DataFrame(['Ws', 'W', 'R_I', 'nFrame', 'Tfb'], columns=['variable'])
	buildingValue 	= pd.DataFrame([2227.5, 3037.5, row[7], 2, 0.735], columns=['value'])

	buildingIndex 	= buildingIndex.join(buildingValue)
	buildingIndex.to_csv('./inputs/buildingInput.csv', index=False)

	# for each input file, run all GMs in the database
	for ind in gmDatabase.index:

		filename 	= str(gmDatabase['filename'][ind])
		filename 	= filename.replace('.AT2', '')						# remove extension from file name
		defFactor 	= float(gmDatabase['scaleFactor'][ind])
	 												
		runStatus 	= eq.runGM(filename, defFactor)						# perform analysis (superStructDesign and buildModel imported within)

		thisRun 	= postprocessing.failurePostprocess(filename, defFactor, runStatus)		# add run results to holder df
		resultsDf 	= pd.concat([thisRun,resultsDf], sort=False)

		resultsDf.to_csv('./sessionOut/sessionSummary.csv', index=False) 	# for now, write every iteration to prevent data loss