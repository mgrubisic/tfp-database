############################################################################
#               Damage estimation of TFP database

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: November 2022

# Description:  Pelicun tool to predict component damage and loss in the 
# initial database

# Open issues:  (1) 

############################################################################

# import helpful packages for numerical analysis
import sys

import numpy as np

import pandas as pd
idx = pd.IndexSlice
pd.options.display.max_rows = 30

import pprint

# and for plotting
from plotly import graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

# and import pelicun classes and methods
from pelicun.base import convert_to_MultiIndex
from pelicun.assessment import Assessment

import warnings
warnings.filterwarnings('ignore')

#%%

# Prepare demand data set to match format
all_demands = pd.read_csv('demand_data.csv', index_col=None,header=None).transpose()
all_demands.columns = all_demands.loc[0]
all_demands = all_demands.iloc[1:, :]
all_demands.columns = all_demands.columns.fillna('EDP')

all_demands = all_demands.set_index('EDP', drop=True)

run_idx = 14
#run_idx = 324
raw_demands = all_demands[['Units', str(run_idx)]]
raw_demands.columns = ['Units', 'Value']
raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
raw_demands.index.names = ['type','loc','dir']

full_isolation_data = pd.read_csv('full_isolation_data.csv', index_col=None)
run_data = full_isolation_data.loc[run_idx]

#%%

# initialize a pelicun Assessment
PAL = Assessment({
    "PrintLog": True, 
    "Seed": 985,
    "Verbose": False,
    "DemandOffset": {"PFA": 0, "PFV": 0}
})

#%%

demands = raw_demands

demands.insert(1, 'Family',"deterministic")


# distribution parameters  - - - - - - - - - - - - - - -
# pelicun uses generic parameter names to handle various distributions within the same data structure
# we need to rename the parameter columns as follows:
# median -> theta_0
# log_std -> theta_1
demands.rename(columns = {'Value': 'Theta_0'}, inplace=True)

demands

#%%

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

demand_sample_ext[('PID_all',0,1)] = demand_sample_ext[[('PID','1','1'),
                                                        ('PID','2','1'),
                                                        ('PID','3','1')]].max(axis=1)
#%%

# add units to the data 
demand_sample_ext.T.insert(0, 'Units',"")

# PFA and SA are in "g" in this example, while PID and RID are "rad"
demand_sample_ext.loc['Units', ['PFA', 'SA_Tm']] = 'g'
demand_sample_ext.loc['Units',['PID', 'PID_all', 'RID']] = 'rad'
demand_sample_ext.loc['Units',['PFV']] = 'inps'


PAL.demand.load_sample(demand_sample_ext)

#%%

# load the component configuration
cmp_marginals = pd.read_csv('cmp_marginals.csv', index_col=0)

print("...")
cmp_marginals.tail(10)



#%% get the structural contents for the moment frame

P58_data = PAL.get_default_data('fragility_DB_FEMA_P58_2nd')
P58_metadata = PAL.get_default_metadata('fragility_DB_FEMA_P58_2nd')

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

structural_components = get_structural_cmp_MF(run_data, P58_metadata)
cmp_marginals = pd.concat([structural_components, cmp_marginals], axis=0)
#%%
# to make the convenience keywords work in the model, 
# we need to specify the number of stories
PAL.stories = 3

# now load the model into Pelicun
PAL.asset.load_cmp_model({'marginals': cmp_marginals})

# let's take a look at the generated marginal parameters
PAL.asset.cmp_marginal_params.loc['B.20.22.001',:]

#%%

# Generate the component quantity sample
PAL.asset.generate_cmp_sample()

# get the component quantity sample - again, use the save function to convert units
cmp_sample = PAL.asset.save_cmp_sample()

cmp_sample.describe()

#%%

# review the damage model - in this example: fragility functions
P58_data = PAL.get_default_data('fragility_DB_FEMA_P58_2nd')


# print(P58_data['Incomplete'].sum(),' incomplete component fragility definitions')

# note that we drop the last three components here (excessiveRID, irreparable, and collapse) 
# because they are not part of P58
cmp_list = cmp_marginals.index.unique().values[:-3]

P58_data_for_this_assessment = P58_data.loc[cmp_list,:].sort_values('Incomplete', ascending=False)

additional_fragility_db = P58_data_for_this_assessment.loc[
    P58_data_for_this_assessment['Incomplete'] == 1].sort_index() 

# additional_fragility_db

# P58_metadata = PAL.get_default_metadata('fragility_DB_FEMA_P58_2nd')

# # # add missing components
# # pprint.pprint(P58_metadata['C.20.11.001a'])

# # C.20.11.001a - Flexible stair with seismic interstory slip joint.  Steel prefab,
# # stringers, steel or concrete filled pan treads
# # placeholders
# additional_fragility_db.loc['C.20.11.001b',('LS1','Theta_0')] = 0.017 # rads
# additional_fragility_db.loc['C.20.11.001b',('LS1','Theta_1')] = 0.5

# additional_fragility_db.loc['C.20.11.001b',('LS2','Theta_0')] = 0.02 # rads
# additional_fragility_db.loc['C.20.11.001b',('LS2','Theta_1')] = 0.5

# additional_fragility_db.loc['C.20.11.001b',('LS3','Theta_0')] = 0.05 # rads
# additional_fragility_db.loc['C.20.11.001b',('LS3','Theta_1')] = 0.5

#%%

# irreparable damage
# this is based on the default values in P58
additional_fragility_db.loc[
    'excessiveRID', [('Demand','Directional'),
                    ('Demand','Offset'),
                    ('Demand','Type'), 
                    ('Demand','Unit')]] = [1, 0, 'Residual Interstory Drift Ratio', 'rad']   

additional_fragility_db.loc[
    'excessiveRID', [('LS1','Family'),
                    ('LS1','Theta_0'),
                    ('LS1','Theta_1')]] = ['lognormal', 0.01, 0.3]   

additional_fragility_db.loc[
    'irreparable', [('Demand','Directional'),
                    ('Demand','Offset'),
                    ('Demand','Type'), 
                    ('Demand','Unit')]] = [1, 0, 'Peak Spectral Acceleration|Tm', 'g']   


# a very high capacity is assigned to avoid damage from demands
additional_fragility_db.loc[
    'irreparable', ('LS1','Theta_0')] = 1e10 

# additional_fragility_db.loc[
#     'irreparable', [('LS1','Family'),
#                     ('LS1','Theta_0'),
#                     ('LS1','Theta_1')]] = ['lognormal', 0.02, 0.3]   

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

sa_judg = calculate_collapse_SaT1(run_data)

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
additional_fragility_db.loc[
    'collapse', [('Demand','Directional'),
                    ('Demand','Offset'),
                    ('Demand','Type'), 
                    ('Demand','Unit')]] = [1, 0, 'Peak Interstory Drift Ratio|all', 'rad']   

# use judgment method, apply 0.6 variance (FEMA P58 ch. 6)
additional_fragility_db.loc[
    'collapse', [('LS1','Family'),
                  ('LS1','Theta_0'),
                  ('LS1','Theta_1')]] = ['lognormal', 0.05, 0.3]

# We set the incomplete flag to 0 for the additional components
additional_fragility_db['Incomplete'] = 0


#%%

# load fragility data
PAL.damage.load_damage_model([
    additional_fragility_db,  # This is the extra fragility data we've just created
    'PelicunDefault/fragility_DB_FEMA_P58_2nd.csv' # and this is a table with the default P58 data    
])

#%%

# damage process: see example file# 

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

# dmg_process = {
#     "1_collapse": {
#         "DS1": "ALL_NA"
#     }
# }


#%%

# Now we can run the calculation
#PAL.damage.calculate(dmg_process=dmg_process)#, block_batch_size=100) #- for large calculations
PAL.damage.calculate(dmg_process=dmg_process)#, block_batch_size=100)
#%%

# Damage estimates
damage_sample = PAL.damage.save_sample()

print("Size of damage results: ", sys.getsizeof(damage_sample)/1024/1024, "MB")

component = 'B.20.22.001'
damage_sample.describe([0.1, 0.5, 0.9]).T.loc[component,:].head(30)

#%% 

# damage plots
dmg_plot = damage_sample.loc[:, component].groupby(level=[0,2], axis=1).sum().T

px.bar(x=dmg_plot.index.get_level_values(1), y=dmg_plot.mean(axis=1), 
       color=dmg_plot.index.get_level_values(0),
       barmode='group',
       labels={
           'x':'Damage State',
           'y':'Component Quantity [ft2]',
           'color': 'Floor'
       },
       title=f'Mean Quantities of component {component} in each Damage State',
       height=500
      )

#%%

# losses - map consequences to damage

# let us prepare the map based on the component list

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

loss_map

#%%

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

# print(P58_data_for_this_assessment['Incomplete'].sum(), ' components have incomplete consequence models assigned.')

# initialize the dataframe
additional_consequences = pd.DataFrame(
    columns = pd.MultiIndex.from_tuples([('Incomplete',''), 
                                         ('Quantity','Unit'), 
                                         ('DV', 'Unit'), 
                                         ('DS1', 'Theta_0')]),
    index=pd.MultiIndex.from_tuples([('replacement','Cost'), 
                                     ('replacement','Time')])
)



# additional_consequences.loc['E.20.22.112a',('DS1','Theta_0')] = 1.229*0.0254 # pfv
# additional_consequences.loc['E.20.22.112a',('DS1','Theta_1')] = 0.5*0.0254

# add the data about replacement cost and time

# use PACT
# assume $250/sf
# assume 40% of replacement cost is labor, $680/worker-day for SF Bay Area
replacement_cost = 250.0*90.0*90.0*4
replacement_time = replacement_cost*0.4/680.0
additional_consequences.loc[('replacement', 'Cost')] = [0, '1 EA', 'USD_2011', replacement_cost]
additional_consequences.loc[('replacement', 'Time')] = [0, '1 EA', 'worker_day', replacement_time]  

additional_consequences

#%%

# Load the loss model to pelicun
PAL.bldg_repair.load_model(
    [additional_consequences, incomplete_cmp,
      "PelicunDefault/bldg_repair_DB_FEMA_P58_2nd.csv"], 
    loss_map)

# PAL.bldg_repair.load_model(
#     [additional_consequences,
#       "PelicunDefault/bldg_repair_DB_FEMA_P58_2nd.csv"], 
#     loss_map)

#%%

# and run the calculations
PAL.bldg_repair.calculate()

#%%

loss_sample = PAL.bldg_repair.sample

print("Size of repair cost & time results: ", sys.getsizeof(loss_sample)/1024/1024, "MB")

# loss_sample['COST']['B.20.22.001'].groupby(level=[0,2,3],axis=1).sum().describe([0.1, 0.5, 0.9]).T

#%%
loss_plot = loss_sample.groupby(level=[0, 2], axis=1).sum()['COST'].iloc[:, :-2]

# we add 100 to the loss values to avoid having issues with zeros when creating a log plot
loss_plot += 100

fig = px.box(y=np.tile(loss_plot.columns, loss_plot.shape[0]), 
       x=loss_plot.values.flatten(), 
       color = [c[0] for c in loss_plot.columns]*loss_plot.shape[0],
       orientation = 'h',
       labels={
           'x':'Aggregate repair cost [2011 USD]',
           'y':'Component ID',
           'color': 'Component Group'
       },
       title=f'Range of repair cost realizations by component type',
       log_x=True,
       height=1500)

fig.update_layout( # customize font and legend orientation & position
    font=dict(size=28)
    )

# fig.show()
#%%


loss_plot = loss_sample['COST'].groupby('loc', axis=1).sum().describe([0.1,0.5,0.9]).iloc[:, 1:]

fig = px.pie(values=loss_plot.loc['mean'],
       names=[f'floor {c}' if int(c)<5 else 'roof' for c in loss_plot.columns],
       title='Contribution of each floor to the average non-collapse repair costs',
       height=500,
       hole=0.4
      )

fig.update_traces(textinfo='percent+label')


#%%


loss_plot = loss_sample['COST'].groupby(level=[1], axis=1).sum()

loss_plot['repairable'] = loss_plot.iloc[:,:-2].sum(axis=1)
loss_plot = loss_plot.iloc[:,-3:]

px.bar(x=loss_plot.columns, 
       y=loss_plot.describe().loc['mean'],
       labels={
           'x':'Damage scenario',
           'y':'Average repair cost'
       },
       title=f'Contribution to average losses from the three possible damage scenarios',
       height=400
      )


# **Aggregate losses**
# 
# Aggregating losses for repair costs is straightforward, but repair times are less trivial. Pelicun adopts the method from FEMA P-58 and provides two bounding values for aggregate repair times:
# - **parallel** assumes that repairs are conducted in parallel across locations. In each location, repairs are assumed to be sequential. This translates to aggregating component repair times by location and choosing the longest resulting aggregate value across locations.
# - **sequential** assumes repairs are performed sequentially across locations and within each location. This translates to aggregating component repair times across the entire building.
# 
# The parallel option is considered a lower bound and the sequential is an upper bound of the real repair time. Pelicun automatically calculates both options for all (i.e., not only FEMA P-58) analyses.

#%%


agg_DF = PAL.bldg_repair.aggregate_losses()

agg_DF.describe([0.1, 0.5, 0.9])


#%%

# filter only the repairable cases
agg_DF_plot = agg_DF.loc[agg_DF['repair_cost'] < 8e6]

px.scatter(x=agg_DF_plot[('repair_time','sequential')],
           y=agg_DF_plot[('repair_time','parallel')], 
           opacity=0.1,
           marginal_x ='histogram', marginal_y='histogram',
           labels={
               'x':'Sequential repair time [worker-days]',
               'y':'Parallel repair time [worker-days]',
           },
           title=f'Two bounds of repair time conditioned on repairable damage',
           height=750, width=750)

px.scatter(x=agg_DF_plot[('repair_time','sequential')],
           y=agg_DF_plot['repair_cost'], 
           opacity=0.1,
           marginal_x ='histogram', marginal_y='histogram',
           labels={
               'x':'Sequential repair time [worker-days]',
               'y':'Repair cost [USD]',
           },
           title=f'Repair time and cost',
           height=750, width=750)