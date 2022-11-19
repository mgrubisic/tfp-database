############################################################################
#               Demand file (pelicun)

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: October 2022

# Description:  Utility used to generate demand files (EDP) in the PBE framework

# Open issues:  (1) requires expansion to generalized multifloor
#               (2) check acceleration units

############################################################################



def get_demand_data(isol_data):

    EDP_data = isol_data[['accMax1', 'accMax2', 'accMax3', 
        'driftMax1', 'driftMax2', 'driftMax3',
        'accMax1', 'accMax2', 'accMax3', 
        'driftMax1', 'driftMax2', 'driftMax3']]

    #1-type-floor-direction
    EDP_data.columns = ['1-PFA-1-1', '1-PFA-2-1', '1-PFA-3-1',
        '1-PID-1-1', '1-PID-2-1', '1-PID-3-1',
        '1-PFA-1-2', '1-PFA-2-2', '1-PFA-3-2',
        '1-PID-1-2', '1-PID-2-2', '1-PID-3-2']

    return(EDP_data)

import pandas as pd
data = pd.read_csv('./sessionOut/sessionSummary_PID-PFA.csv', sep=',')
data = data[data.runFailed == 0]
edp = get_demand_data(data)
edp.to_csv('./sessionOut/engineering_demands_TFP.csv', index=False)