############################################################################
#               Damage estimation of TFP database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: November 2022

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

import warnings
warnings.filterwarnings('ignore')

###########################################################################
# ASSEMBLE STRUCTURAL COMPONENTS
###########################################################################

def get_structural_cmp_MF(run_info, metadata):
    cmp_strct = pd.DataFrame()

    beam_str = run_info['beam']
    col_str = run_info['col']
    
    col_wt = float(col_str.split('X',1)[1])
    beam_depth = float(beam_str.split('X',1)[0].split('W',1)[1])
    
    # bolted shear tab gravity, assume 1 per every 10 ft span in one direction
    cur_cmp = 'B.10.31.001'
    cmp_strct.loc[
        cur_cmp, ['Units', 'Location', 'Direction',
                        'Theta_0', 'Theta_1', 'Family',
                        'Blocks', 'Comment']] = ['ea', '1--3', '0',
                                                 9*6, np.nan, np.nan,
                                                 9*6, metadata[cur_cmp]['Description']]
    # column base plates
    if col_wt < 150.0:
        cur_cmp = 'B.10.31.011a'
    elif col_wt > 300.0:
        cur_cmp = 'B.10.31.011c'
    else:
        cur_cmp = 'B.10.31.011b'
    
    cmp_strct.loc[
        cur_cmp, ['Units', 'Location', 'Direction',
                        'Theta_0', 'Theta_1', 'Family',
                        'Blocks', 'Comment']] = ['ea', '1', '1,2',
                                                 4*4, np.nan, np.nan,
                                                 4*4, metadata[cur_cmp]['Description']]
    # assume no splice needed in column (39 ft)
    
    # moment connection, one beam (all roof beams are < 27.0)
    if beam_depth <= 27.0:
        cur_cmp = 'B.10.35.021'
        cmp_strct.loc[
            cur_cmp, ['Units', 'Location', 'Direction',
                            'Theta_0', 'Theta_1', 'Family',
                            'Blocks', 'Comment']] = ['ea', '1--3', '1,2',
                                                     8, np.nan, np.nan,
                                                     8, metadata[cur_cmp]['Description']]
    else:
        cur_cmp = 'B.10.35.022'
        cmp_strct.loc[
            cur_cmp, ['Units', 'Location', 'Direction',
                            'Theta_0', 'Theta_1', 'Family',
                            'Blocks', 'Comment']] = ['ea', '1--2', '1,2',
                                                     8, np.nan, np.nan,
                                                     8, metadata[cur_cmp]['Description']]
        cur_cmp = 'B.10.35.021'
        cmp_strct.loc[
            cur_cmp, ['Units', 'Location', 'Direction',
                            'Theta_0', 'Theta_1', 'Family',
                            'Blocks', 'Comment']] = ['ea', '3', '1,2',
                                                     8, np.nan, np.nan,
                                                     8, metadata[cur_cmp]['Description']]
                                                     
    # moment connection, two beams (all roof beams are < 27.0)
    if beam_depth <= 27.0:
        cur_cmp = 'B.10.35.031'
        cmp_strct.loc[
            cur_cmp, ['Units', 'Location', 'Direction',
                            'Theta_0', 'Theta_1', 'Family',
                            'Blocks', 'Comment']] = ['ea', '1--3', '1,2',
                                                     8, np.nan, np.nan,
                                                     8, metadata[cur_cmp]['Description']]
    else:
        cur_cmp = 'B.10.35.032'
        cmp_strct.loc[
            cur_cmp, ['Units', 'Location', 'Direction',
                            'Theta_0', 'Theta_1', 'Family',
                            'Blocks', 'Comment']] = ['ea', '1--2', '1,2',
                                                     8, np.nan, np.nan,
                                                     8, metadata[cur_cmp]['Description']]
        cur_cmp = 'B.10.35.031'
        cmp_strct.loc[
            cur_cmp, ['Units', 'Location', 'Direction',
                            'Theta_0', 'Theta_1', 'Family',
                            'Blocks', 'Comment']] = ['ea', '3', '1,2',
                                                     8, np.nan, np.nan,
                                                     8, metadata[cur_cmp]['Description']]
        
    return(cmp_strct)

def estimate_damage(raw_demands, run_data, cmp_marginals):
    
    # initialize, no printing outputs, offset fixed with current components
    PAL = Assessment({
        "PrintLog": False, 
        "Seed": 985,
        "Verbose": False,
        "DemandOffset": {"PFA": 0, "PFV": 0}
    })
    
    ###########################################################################
    # DEMANDS
    ###########################################################################
    # specify deterministic demands
    demands = raw_demands
    demands.insert(1, 'Family',"deterministic")
    demands.rename(columns = {'Value': 'Theta_0'}, inplace=True)
    
    # prepare a correlation matrix that represents perfect correlation
    ndims = demands.shape[0]
    demand_types = demands.index 

    perfect_CORR = pd.DataFrame(
        np.ones((ndims, ndims)),
        columns = demand_types,
        index = demand_types)

    # load the demand model
    PAL.demand.load_model({'marginals': demands,
                           'correlation': perfect_CORR})

    # generate demand sample
    n_sample = 10000
    PAL.demand.generate_sample({"SampleSize": n_sample})

    # extract the generated sample
    # Note that calling the save_sample() method is better than directly pulling the 
    # sample attribute from the demand object because the save_sample method converts
    # demand units back to the ones you specified when loading in the demands.
    demand_sample = PAL.demand.save_sample()


    # get residual drift estimates 
    delta_y = 0.0075 # found from typical pushover curve for structure
    PID = demand_sample['PID']

    RID = PAL.demand.estimate_RID(PID, {'yield_drift': delta_y}) 

    # and join them with the demand_sample
    demand_sample_ext = pd.concat([demand_sample, RID], axis=1)

    # add spectral acceleration at fundamental period (BEARING EFFECTIVE PERIOD)
    # does Sa(T) characterize this well considering impact? for now, try using peak acceleration
    # demand_sample_ext[('SA_Tm',0,1)] = max(run_data['accMax0'],
    #                                        run_data['accMax1'],
    #                                        run_data['accMax2'],
    #                                        run_data['accMax3'])

    demand_sample_ext[('SA_Tm',0,1)] = run_data['GMSTm']
    
    demand_sample_ext[('PID_all',0,1)] = demand_sample_ext[[('PID','1','1'),
                                                            ('PID','2','1'),
                                                            ('PID','3','1')]].max(axis=1)
    
    # add units to the data 
    demand_sample_ext.T.insert(0, 'Units',"")

    # PFA and SA are in "g" in this example, while PID and RID are "rad"
    demand_sample_ext.loc['Units', ['PFA', 'SA_Tm']] = 'g'
    demand_sample_ext.loc['Units',['PID', 'PID_all', 'RID']] = 'rad'
    demand_sample_ext.loc['Units',['PFV']] = 'inps'


    PAL.demand.load_sample(demand_sample_ext)
    
    ###########################################################################
    # COMPONENTS
    ###########################################################################
    
    # generate structural components and join with NSCs
    P58_metadata = PAL.get_default_metadata('fragility_DB_FEMA_P58_2nd')
    cmp_structural = get_structural_cmp_MF(run_data, P58_metadata)
    cmp_marginals = pd.concat([cmp_structural, cmp_marginals], axis=0)
    
    # to make the convenience keywords work in the model, 
    # we need to specify the number of stories
    PAL.stories = 3

    # now load the model into Pelicun
    PAL.asset.load_cmp_model({'marginals': cmp_marginals})
    
    # Generate the component quantity sample
    PAL.asset.generate_cmp_sample()

    # get the component quantity sample - again, use the save function to convert units
    cmp_sample = PAL.asset.save_cmp_sample()
    
    # review the damage model - in this example: fragility functions
    P58_data = PAL.get_default_data('fragility_DB_FEMA_P58_2nd')

    # note that we drop the last three components here (excessiveRID, irreparable, and collapse) 
    # because they are not part of P58
    cmp_list = cmp_marginals.index.unique().values[:-3]

    P58_data_for_this_assessment = P58_data.loc[cmp_list,:].sort_values('Incomplete', ascending=False)

    additional_fragility_db = P58_data_for_this_assessment.loc[
        P58_data_for_this_assessment['Incomplete'] == 1].sort_index() 
    
    # add demand for the replacement criteria
    # irreparable damage
    # this is based on the default values in P58
    additional_fragility_db.loc[
        'excessiveRID', [('Demand','Directional'),
                        ('Demand','Offset'),
                        ('Demand','Type'), 
                        ('Demand','Unit')]] = [1, 
                                               0, 
                                               'Residual Interstory Drift Ratio',
                                               'rad']   

    additional_fragility_db.loc[
        'excessiveRID', [('LS1','Family'),
                        ('LS1','Theta_0'),
                        ('LS1','Theta_1')]] = ['lognormal', 0.01, 0.3]   

    additional_fragility_db.loc[
        'irreparable', [('Demand','Directional'),
                        ('Demand','Offset'),
                        ('Demand','Type'), 
                        ('Demand','Unit')]] = [1,
                                               0,
                                               'Peak Spectral Acceleration|Tm',
                                               'g']   


    # a very high capacity is assigned to avoid damage from demands
    # this will trigger on excessiveRID instead
    additional_fragility_db.loc[
        'irreparable', ('LS1','Theta_0')] = 1e10 

    

    # sa_judg = calculate_collapse_SaT1(run_data)

    # capacity is assigned based on the example in the FEMA P58 background documentation
    # additional_fragility_db.loc[
    #     'collapse', [('Demand','Directional'),
    #                     ('Demand','Offset'),
    #                     ('Demand','Type'), 
    #                     ('Demand','Unit')]] = [1, 0, 'Peak Spectral Acceleration|Tm', 'g']   

    # # use judgment method, apply 0.6 variance (FEMA P58 ch. 6)
    # additional_fragility_db.loc[
    #     'collapse', [('LS1','Family'),
    #                   ('LS1','Theta_0'),
    #                   ('LS1','Theta_1')]] = ['lognormal', sa_judg, 0.6]  

    # collapse capacity is assumed 5% interstory drift across any floor
    # Mean provided by Masroor and Mosqueda
    # Std from Yun and Hamburger (2002)
    
    # TODO: stats validation, find justification for this 
    # we can define a lognormal distribution that results in a PID of 5% having
    # 90% collapse rate
    # Yun and Hamburger has beta (logarithmic stdev) value of 0.3 for 
    # 3-story global collapse drift, lowered by 0.05 if nonlin dynamic anly
    from math import log, exp
    beta_drift = 0.25
    mean_log_drift = exp(log(0.05) - beta_drift*1.2816) # 1.2816 is inverse normCDF of 0.90
    additional_fragility_db.loc[
        'collapse', [('Demand','Directional'),
                        ('Demand','Offset'),
                        ('Demand','Type'), 
                        ('Demand','Unit')]] = [1, 0, 'Peak Interstory Drift Ratio|all', 'rad']   

    additional_fragility_db.loc[
        'collapse', [('LS1','Family'),
                      ('LS1','Theta_0'),
                      ('LS1','Theta_1')]] = ['lognormal', mean_log_drift, beta_drift]

    # We set the incomplete flag to 0 for the additional components
    additional_fragility_db['Incomplete'] = 0
    
    # load fragility data
    PAL.damage.load_damage_model([
        additional_fragility_db,  # This is the extra fragility data we've just created
        'PelicunDefault/fragility_DB_FEMA_P58_2nd.csv' # and this is a table with the default P58 data    
    ])
    
    ### 3.3.5 Damage Process
    # 
    # Damage processes are a powerful new feature in Pelicun 3. 
    # They are used to connect damages of different components in the performance model 
    # and they can be used to create complex cascading damage models.
    # 
    # The default FEMA P-58 damage process is farily simple. The process below can be interpreted as follows:
    # * If Damage State 1 (DS1) of the collapse component is triggered (i.e., the building collapsed), 
    # then damage for all other components should be cleared from the results. 
    # This considers that component damages (and their consequences) in FEMA P-58 are conditioned on no collapse.

    # * If Damage State 1 (DS1) of any of the excessiveRID components is triggered 
    # (i.e., the residual drifts are larger than the prescribed capacity on at least one floor),
    # then the irreparable component should be set to DS1.

    # FEMA P58 uses the following process:
    dmg_process = {
        "1_collapse": {
            "DS1": "ALL_NA"
        },
        "2_excessiveRID": {
            "DS1": "irreparable_DS1"
        }
    }
    
    ###########################################################################
    # DAMAGE
    ###########################################################################
    
    print('Damage estimation...')
    # Now we can run the calculation
    PAL.damage.calculate(dmg_process=dmg_process)#, block_batch_size=100)
    
    # Damage estimates
    damage_sample = PAL.damage.save_sample()
    
    ###########################################################################
    # LOSS
    ###########################################################################
    
    # we need to prepend 'DMG-' to the component names to tell pelicun to look for the damage of these components
    drivers = [f'DMG-{cmp}' for cmp in cmp_marginals.index.unique()]
    drivers = drivers[:-3]+drivers[-2:]

    # we are looking at repair consequences in this example
    # the components in P58 have consequence models under the same name
    loss_models = cmp_marginals.index.unique().tolist()[:-3]

    # We will define the replacement consequence in the following cell.
    loss_models+=['replacement',]*2

    # Assemble the DataFrame with the mapping information
    # The column name identifies the type of the consequence model.
    loss_map = pd.DataFrame(loss_models, columns=['BldgRepair'], index=drivers)
    
    # load the consequence models
    P58_data = PAL.get_default_data('bldg_repair_DB_FEMA_P58_2nd')

    # group E
    incomplete_cmp = pd.DataFrame(
        columns = pd.MultiIndex.from_tuples([('Incomplete',''), 
                                             ('Quantity','Unit'), 
                                             ('DV', 'Unit'), 
                                             ('DS1','Theta_0'),
                                             ('DS1','Theta_1'),
                                             ('DS1','Family'),]),
        index=pd.MultiIndex.from_tuples([('E.20.22.102a','Cost'), 
                                         ('E.20.22.102a','Time'),
                                         ('E.20.22.112a','Cost'), 
                                         ('E.20.22.112a','Time')])
    )

    incomplete_cmp.loc[('E.20.22.102a', 'Cost')] = [0, '1 EA', 'USD_2011',
                                                 '190.0, 150.0|1,5', 0.35, 'lognormal']
    incomplete_cmp.loc[('E.20.22.102a', 'Time')] = [0, '1 EA', 'worker_day',
                                                 0.02, 0.5, 'lognormal']

    incomplete_cmp.loc[('E.20.22.112a', 'Cost')] = [0, '1 EA', 'USD_2011',
                                                 '110.0, 70.0|1,5', 0.35, 'lognormal']
    incomplete_cmp.loc[('E.20.22.112a', 'Time')] = [0, '1 EA', 'worker_day',
                                                 0.02, 0.5, 'lognormal']

    # get the consequences used by this assessment
    P58_data_for_this_assessment = P58_data.loc[loss_map['BldgRepair'].values[:-5],:]

    # initialize the dataframe
    additional_consequences = pd.DataFrame(
        columns = pd.MultiIndex.from_tuples([('Incomplete',''), 
                                             ('Quantity','Unit'), 
                                             ('DV', 'Unit'), 
                                             ('DS1', 'Theta_0')]),
        index=pd.MultiIndex.from_tuples([('replacement','Cost'), 
                                         ('replacement','Time')])
    )
    
    # add the data about replacement cost and time

    # use PACT
    # assume $250/sf
    # assume 40% of replacement cost is labor, $680/worker-day for SF Bay Area
    replacement_cost = 250.0*90.0*90.0*4
    replacement_time = replacement_cost*0.4/680.0
    additional_consequences.loc[('replacement', 'Cost')] = [0, '1 EA',
                                                            'USD_2011',
                                                            replacement_cost]
    additional_consequences.loc[('replacement', 'Time')] = [0, '1 EA',
                                                            'worker_day',
                                                            replacement_time]
    
    # Load the loss model to pelicun
    PAL.bldg_repair.load_model(
        [additional_consequences, incomplete_cmp,
          "PelicunDefault/bldg_repair_DB_FEMA_P58_2nd.csv"], 
        loss_map)
    
    # and run the calculations
    print('Loss estimation...')
    PAL.bldg_repair.calculate()
    
    # loss estimates
    loss_sample = PAL.bldg_repair.sample
    
    # group components and ensure that all components and replacement are present
    loss_by_cmp = loss_sample.groupby(level=[0, 2], axis=1).sum()['COST']
    for cmp_grp in list(cmp_list):
        if cmp_grp not in list(loss_by_cmp.columns):
            loss_by_cmp[cmp_grp] = 0
            
    # grab replacement cost and convert to instances, fill with zeros if needed
    replacement_instances = pd.DataFrame()
    try:
        replacement_instances['collapse'] = loss_by_cmp['collapse']/replacement_cost
    except KeyError:
        loss_by_cmp['collapse'] = 0
        replacement_instances['collapse'] = pd.DataFrame(np.zeros((10000, 1)))
    try:
        replacement_instances['irreparable'] = loss_by_cmp['irreparable']/replacement_cost
    except KeyError:
        loss_by_cmp['irreparable'] = 0
        replacement_instances['irreparable'] = pd.DataFrame(np.zeros((10000, 1)))
    replacement_instances = replacement_instances.astype(int)
            
    # summarize by groups
    loss_groups = pd.DataFrame()
    loss_groups['B'] = loss_by_cmp[[
        col for col in loss_by_cmp.columns if col.startswith('B')]].sum(axis=1)
    loss_groups['C'] = loss_by_cmp[[
        col for col in loss_by_cmp.columns if col.startswith('C')]].sum(axis=1)
    loss_groups['D'] = loss_by_cmp[[
        col for col in loss_by_cmp.columns if col.startswith('D')]].sum(axis=1)
    loss_groups['E'] = loss_by_cmp[[
        col for col in loss_by_cmp.columns if col.startswith('E')]].sum(axis=1)
    
    # only summarize repair cost from non-replacement cases
    loss_groups = loss_groups.loc[
        (replacement_instances['collapse'] == 0) & (replacement_instances['irreparable'] == 0)]
    
    # TODO: these two conditions are mutually exclusive
    collapse_freq = replacement_instances['collapse'].sum(axis=0)/n_sample
    irreparable_freq = replacement_instances['irreparable'].sum(axis=0)/n_sample
    
    # this returns NaN if collapse/irreparable is 100%
    loss_groups = loss_groups.describe()
    
    # aggregate
    agg_DF = PAL.bldg_repair.aggregate_losses()
    
    return(cmp_sample, damage_sample, loss_sample, loss_groups, agg_DF,
           collapse_freq, irreparable_freq)

#%% prepare whole set of runs

full_isolation_data = pd.read_csv('full_isolation_data.csv', index_col=None)


# load the component configuration
cmp_marginals = pd.read_csv('cmp_marginals.csv', index_col=0)

# Prepare demand data set to match format
all_demands = pd.read_csv('demand_data.csv', index_col=None,header=None).transpose()

all_demands.columns = all_demands.loc[0]
all_demands = all_demands.iloc[1:, :]
all_demands.columns = all_demands.columns.fillna('EDP')

all_demands = all_demands.set_index('EDP', drop=True)

# run_idx = 45
# #run_idx = 324
# raw_demands = all_demands[['Units', str(run_idx)]]
# raw_demands.columns = ['Units', 'Value']
# raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
# raw_demands.index.names = ['type','loc','dir']

# run_data = full_isolation_data.loc[run_idx]

#%% estimate loss for set

all_losses = []
loss_cmp_group = []
col_list = []
irr_list = []

# for run_idx in range(3):
for run_idx in range(len(full_isolation_data)):
    run_data = full_isolation_data.loc[run_idx]
    
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

#%% flatten data

loss_df = pd.read_csv(loss_file, header=[0,1])

loss_header = ['cost_mean', 'cost_std', 'cost_min',
               'cost_10%', 'cost_50%', 'cost_90%', 'cost_max',
               'time_l_mean', 'time_l_std', 'time_l_min',
               'time_l_10%', 'time_l_50%', 'time_l_90%', 'time_l_max',
               'time_u_mean', 'time_u_std', 'time_u_min',
               'time_u_10%', 'time_u_50%', 'time_u_90%', 'time_u_max']

all_rows = []

for row_idx in range(len(loss_df)):
    if row_idx % 8 == 0:
        # get the block with current run, drop the 'Count'
        run_df = loss_df[row_idx:row_idx+8]
        run_df = run_df.transpose()
        run_df.columns = run_df.iloc[0]
        run_df = run_df.drop(run_df.index[0])
        new_row = pd.concat([run_df.iloc[0], run_df.iloc[1], run_df.iloc[2]])
        new_row = new_row.drop(new_row.index[0])
        
        all_rows.append(new_row)
        
loss_df_data = pd.concat(all_rows, axis=1).T
loss_df_data.columns = loss_header

loss_df_data['collapse_freq'] = col_list
loss_df_data['irreparable_freq'] = irr_list
loss_df_data['replacement_freq'] = [x + y for x, y in zip(col_list, irr_list)]

# loss_df_data.to_csv(loss_file, index=False)
#%%
group_df = pd.read_csv(by_cmp_file, header=0)
group_header = ['B_mean', 'B_std', 'B_min',
               'B_25%', 'B_50%', 'B_75%', 'B_max',
               'C_mean', 'C_std', 'C_min',
               'C_25%', 'C_50%', 'C_75%', 'C_max',
               'D_mean', 'D_std', 'D_min',
               'D_25%', 'D_50%', 'D_75%', 'D_max',
               'E_mean', 'E_std', 'E_min',
               'E_25%', 'E_50%', 'E_75%', 'E_max']

all_rows = []

for row_idx in range(len(group_df)):
    if row_idx % 8 == 0:
        # get the block with current run, drop the 'Count'
        run_df = group_df[row_idx:row_idx+8]
        run_df = run_df.transpose()
        run_df.columns = run_df.iloc[0]
        run_df = run_df.drop(run_df.index[0])
        new_row = pd.concat([run_df.iloc[0], run_df.iloc[1], run_df.iloc[2], run_df.iloc[3]])
        new_row = new_row.drop(new_row.index[0])
        
        all_rows.append(new_row)
        
group_df_data = pd.concat(all_rows, axis=1).T
group_df_data.columns = group_header

all_data = pd.concat([loss_df_data, group_df_data], axis=1)
all_data.to_csv(loss_file, index=False)