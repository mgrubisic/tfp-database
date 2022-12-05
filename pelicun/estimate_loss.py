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

# collapse
def calculate_collapse_SaT1(run_series):
    
    # method described in FEMA P-58 Ch. 6.4
    # problem is we're working with Tm and not fundamental period
    Dm = run_series['DesignDm']
    Tm = run_series['Tm']
    R = run_series['RI']
    kM = (1/386.4)*(2*3.14159/Tm)**2
    
    Vb = Dm * kM
    
    # assumes that structure has identical property in each direction
    Sa_D = Vb*R
    return(4*Sa_D)

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
    PAL.demand.generate_sample({"SampleSize": 10000})

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
    
    # add units to the data 
    demand_sample_ext.T.insert(0, 'Units',"")

    # PFA and SA are in "g" in this example, while PID and RID are "rad"
    demand_sample_ext.loc['Units', ['PFA', 'SA_Tm']] = 'g'
    demand_sample_ext.loc['Units',['PID', 'RID']] = 'rad'
    demand_sample_ext.loc['Units',['PFV']] = 'inps'


    PAL.demand.load_sample(demand_sample_ext)
    
    ###########################################################################
    # COMPONENTS
    ###########################################################################
    
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

    

    sa_judg = calculate_collapse_SaT1(run_data)

    # capacity is assigned based on the example in the FEMA P58 background documentation
    additional_fragility_db.loc[
        'collapse', [('Demand','Directional'),
                        ('Demand','Offset'),
                        ('Demand','Type'), 
                        ('Demand','Unit')]] = [1,
                                               0,
                                               'Peak Spectral Acceleration|Tm',
                                               'g']   

    # use judgment method, apply 0.6 variance (FEMA P58 ch. 6)
    additional_fragility_db.loc[
        'collapse', [('LS1','Family'),
                      ('LS1','Theta_0'),
                      ('LS1','Theta_1')]] = ['lognormal', sa_judg, 0.6]  

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
    
    agg_DF = PAL.bldg_repair.aggregate_losses()
    
    return(cmp_sample, damage_sample, loss_sample, agg_DF)

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
for run_idx in range(2):
    run_data = full_isolation_data.loc[run_idx]
    
    raw_demands = all_demands[['Units', str(run_idx)]]
    raw_demands.columns = ['Units', 'Value']
    raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
    raw_demands.index.names = ['type','loc','dir']
    
    print('========================================')
    print('Estimating loss for run index', run_idx)
    
    cmp, dmg, loss, agg = estimate_damage(raw_demands, run_data, cmp_marginals)
    loss_summary = agg.describe([0.1, 0.5, 0.9])
    cost = loss_summary['repair_cost']['mean']
    time_l = loss_summary[('repair_time', 'parallel')]['mean']
    time_u = loss_summary[('repair_time', 'sequential')]['mean']
    print('Mean repair cost: ', f'${cost:,.2f}')
    print('Mean lower bound repair time: ', f'{time_l:,.2f}', 'worker-days')
    print('Mean upper bound repair time: ', f'{time_u:,.2f}', 'worker-days')
    all_losses.append(loss_summary)
    
pd.concat(all_losses).to_csv('./results/loss_estimate.csv')

#%% flatten data

loss_df = pd.read_csv('./results/loss_estimate.csv', header=[0,1])

loss_header = ['cost_mean', 'cost_std', 'cost_min',
               'cost_10%', 'cost_50%', 'cost_90%', 'cost_max',
               'time_l_mean', 'time_l_std', 'time_l_min',
               'time_l_10%', 'time_l_50%', 'time_l_90%', 'time_l_max',
               'time_u_mean', 'time_u_std', 'time_u_min',
               'time_u_10%', 'time_u_50%', 'time_u_90%', 'time_u_max']

all_rows = []

for row_idx in range(len()):
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

loss_df_data.to_csv('./results/loss_estimate_data.csv', index=False)