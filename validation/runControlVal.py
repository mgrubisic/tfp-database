############################################################################
#             	Run control

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	December 2020

# Description: 	Main script, reworked for validation study
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
import numpy as np
import postprocessing
import eqAnly as eq
import gmSelector

# initialize dataframe as an empty object
resultsDf = None

# filter GMs, then get ground motion database list
gmPath 	= '../groundMotions/PEERNGARecords_Unscaled/'
PEERSummary = 'combinedSearch.csv'
databaseFile = 'gmListVal.csv'

# incremental MCE_R levels
#IDALevel = np.arange(1.0, 2.50, 0.5).tolist()
IDALevel = np.arange(1.0, 2.5, 0.5).tolist()

# read in params
parCsv 	= pd.read_csv('./inputs/bearingInputVal.csv', index_col=None, header=0)
param = dict(zip(parCsv.variable, parCsv.value))

for lvl in IDALevel:

    print('The IDA level is ' + str(lvl) + '.')

    empty_directory('outputs')

    # scale S1 to match MCE_R level
    actualS1 	= param['S1']*lvl
    
    gmDatabase, specAvg = gmSelector.cleanGMs(gmPath, PEERSummary, actualS1, lvl,
        32, 133, 176, 111, 290, 111)
    
    # Run eq analysis for 
    for gmIdx in range(len(gmDatabase)):
    # for gmIdx in range(1):

        # ground motion name, extension removed
        filename    = str(gmDatabase['filename'][gmIdx])
        filename    = filename.replace('.AT2', '')

        # scale factor used, either scaleFactorS1 or scaleFactorSpecAvg
        defFactor   = float(gmDatabase['scaleFactorSpecAvg'][gmIdx])

        # move on to next set if bad friction coeffs encountered 
        # (handled in superStructDesign)
        try:
            runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.005) # perform analysis (superStructDesign and buildModel imported within)
        except ValueError:
            raise ValueError('Bearing solver returned negative friction coefficients. Skipping...')
        except IndexError:
            raise IndexError('SCWB check failed, no shape exists for design. Skipping...')
        if runStatus != 0:
            print('Lowering time step...')
            runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.001)
        if runStatus != 0:
            print('Lowering time step last time...')
            runStatus, Tfb, scaleFactor = eq.runGM(filename, defFactor, 0.0005)
        if runStatus != 0:
            print('Recording run and moving on.')


        # add run results to holder df
        resultsHeader, thisRun  = postprocessing.failurePostprocess(filename, 
            scaleFactor, specAvg, runStatus, Tfb, lvl)

        # if initial run, start the dataframe with headers from postprocessing
        if resultsDf is None:
            resultsDf           = pd.DataFrame(columns=resultsHeader)
        
        # add results onto the dataframe
        resultsDf               = pd.concat([thisRun,resultsDf], sort=False)

gmDatabase.to_csv(gmPath+databaseFile, index=False)
resultsDf.to_csv('./sessionOut/sessionSummary.csv', index=False)

#%% 
# import sys

# # setting path
# sys.path.append('..')

# # importing
# import tmp_cleaner
# from get_demand_data import get_EDP
# databasePath = './sessionOut/'
# databaseFile = 'sessionSummary.csv'

# # clean data and add additional variables
# data = tmp_cleaner.cleanDat(resultsDf)
# pelicunPath = '../pelicun/'
# data.to_csv(pelicunPath+'validation_data.csv', index=False)

# # write into pelicun style EDP
# edp = get_EDP(data)
# edp.to_csv(pelicunPath+'validation_demand_data.csv', index=True)