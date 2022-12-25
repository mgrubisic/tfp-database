############################################################################
#               Validate losses for a single design

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  Pelicun tool to predict component damage and loss in the 
# initial database

# Open issues:  (1) 

############################################################################

import numpy as np

import pandas as pd
idx = pd.IndexSlice
pd.options.display.max_rows = 30

# and import pelicun classes and methods
from pelicun.base import convert_to_MultiIndex
from pelicun.assessment import Assessment

# get other premade methods to estimate loss
from estimate_loss import get_structural_cmp_MF, estimate_damage

import warnings
warnings.filterwarnings('ignore')

#%% prepare whole set of runs

isolation_data = pd.read_csv('validation_data.csv', index_col=None)


# load the component configuration
cmp_marginals = pd.read_csv('cmp_marginals.csv', index_col=0)

# Prepare demand data set to match format
all_demands = pd.read_csv('validation_demand_data.csv',
                          index_col=None,header=None).transpose()

all_demands.columns = all_demands.loc[0]
all_demands = all_demands.iloc[1:, :]
all_demands.columns = all_demands.columns.fillna('EDP')

all_demands = all_demands.set_index('EDP', drop=True).transpose()


# just need the design spec from one row
run_data = isolation_data.loc[1]

# EDP DISTRIBUTION PER LEVEL IMPLEMENTATION
for level in isolation_data['IDALevel'].unique():
    
    # separate the validation set by IDA levels
    row_list = isolation_data['IDALevel'] == level
    
    # get the unit row as well
    row_list = pd.concat([pd.Series([True]), row_list])
    many_demands = all_demands.loc[row_list]
    
    # TODO: clean the EDPs for many demands

    print('========================================')
    print('Estimating loss for level', level)
    
    [cmp, dmg, loss, loss_cmp, agg, 
         collapse_rate, irr_rate] = estimate_damage(many_demands,
                                                run_data,
                                                cmp_marginals,
                                                mode='validation')
    
# TODO: implementation that treats each validation run as deterministic

#%% Task overview

# separate the demands/EDPs per IDA level
# use each IDA level as one-bldg-multiple-EDP run (get distro)
# for each level, estimate single loss distribution

# OR

# separate the demands/EDPs per IDA level
# use each IDA level as n-separate deterministic runs
# for each level get ~50 samples of loss

#%% estimate loss for set

all_losses = []
loss_cmp_group = []
col_list = []
irr_list = []

# for run_idx in range(3):
for run_idx in range(len(isolation_data)):
    run_data = isolation_data.loc[run_idx]
    
    raw_demands = all_demands[['Units', str(run_idx)]]
    raw_demands.columns = ['Units', 'Value']
    raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
    raw_demands.index.names = ['type','loc','dir']
    
    print('========================================')
    print('Estimating loss for run index', run_idx)
    
    [cmp, dmg, loss, loss_cmp, agg, 
         collapse_rate, irr_rate] = estimate_damage(raw_demands,
                                                run_data,
                                                cmp_marginals)
    loss_summary = agg.describe([0.1, 0.5, 0.9])
    cost = loss_summary['repair_cost']['mean']
    time_l = loss_summary[('repair_time', 'parallel')]['mean']
    time_u = loss_summary[('repair_time', 'sequential')]['mean']
    
    print('Mean repair cost: ', f'${cost:,.2f}')
    print('Mean lower bound repair time: ', f'{time_l:,.2f}', 'worker-days')
    print('Mean upper bound repair time: ', f'{time_u:,.2f}', 'worker-days')
    print('Collapse frequency: ', f'{collapse_rate:.2%}')
    print('Irreparable RID frequency: ', f'{irr_rate:.2%}')
    print('Replacement frequency: ', f'{collapse_rate+irr_rate:.2%}')
    all_losses.append(loss_summary)
    loss_cmp_group.append(loss_cmp)
    col_list.append(collapse_rate)
    irr_list.append(irr_rate)
    
loss_file = './results/loss_estimate_data.csv'
by_cmp_file = './results/loss_estimate_by_groups.csv'
pd.concat(all_losses).to_csv(loss_file)
pd.concat(loss_cmp_group).to_csv(by_cmp_file)
