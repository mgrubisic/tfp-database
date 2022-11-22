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

#%%

# Prepare demand data set to match format
all_demands = pd.read_csv('demand_data.csv', index_col=None,header=None).transpose()
all_demands.columns = all_demands.loc[0]
all_demands = all_demands.iloc[1:, :]
all_demands.columns = all_demands.columns.fillna('EDP')

all_demands = all_demands.set_index('EDP', drop=True)


run_idx = 6
raw_demands = all_demands[['Units', str(run_idx)]]
raw_demands.columns = ['Units', 'Value']
raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
raw_demands.index.names = ['type','loc','dir']
#%%

# initialize a pelicun Assessment
PAL = Assessment({
    "PrintLog": True, 
    "Seed": 985,
    "Verbose": False,
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
delta_y = 0.01 # sample value for ductile steel MRF, FEMA P-58 T. C-2
PID = demand_sample['PID']

RID = PAL.demand.estimate_RID(PID, {'yield_drift': delta_y}) 

# and join them with the demand_sample
demand_sample_ext = pd.concat([demand_sample, RID], axis=1)

#%%

# add units to the data 
demand_sample_ext.T.insert(0, 'Units',"")

# PFA and SA are in "g" in this example, while PID and RID are "rad"
demand_sample_ext.loc['Units', ['PFA']] = 'g'
demand_sample_ext.loc['Units',['PID', 'RID']] = 'rad'


PAL.demand.load_sample(demand_sample_ext)

#%%

# load the component configuration
cmp_marginals = pd.read_csv('cmp_marginals.csv', index_col=0)

print("...")
cmp_marginals.tail(10)

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


print(P58_data['Incomplete'].sum(),' incomplete component fragility definitions')

# note that we drop the last three components here (excessiveRID, irreparable, and collapse) 
# because they are not part of P58
cmp_list = cmp_marginals.index.unique().values[:-3]

P58_data_for_this_assessment = P58_data.loc[cmp_list,:].sort_values('Incomplete', ascending=False)

additional_fragility_db = P58_data_for_this_assessment.loc[
    P58_data_for_this_assessment['Incomplete'] == 1].sort_index() 

additional_fragility_db

P58_metadata = PAL.get_default_metadata('fragility_DB_FEMA_P58_2nd')

# add missing components
pprint.pprint(P58_metadata['C.20.11.001a'])

# C.20.11.001a - Flexible stair with seismic interstory slip joint.  Steel prefab,
# stringers, steel or concrete filled pan treads
# placeholders
additional_fragility_db.loc['C.20.11.001a',('LS1','Theta_0')] = 0.017 # rads
additional_fragility_db.loc['C.20.11.001a',('LS1','Theta_1')] = 0.01

additional_fragility_db.loc['C.20.11.001a',('LS2','Theta_0')] = 0.02 # rads
additional_fragility_db.loc['C.20.11.001a',('LS2','Theta_1')] = 0.01

additional_fragility_db.loc['C.20.11.001a',('LS3','Theta_0')] = 0.05 # rads
additional_fragility_db.loc['C.20.11.001a',('LS3','Theta_1')] = 0.01

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
                    ('LS1','Theta_1')]] = ['lognormal', 0.017, 0.01]   

additional_fragility_db.loc[
    'irreparable', [('Demand','Directional'),
                    ('Demand','Offset'),
                    ('Demand','Type'), 
                    ('Demand','Unit')]] = [1, 0, 'Peak Interstory Drift Ratio', 'rad']   


# a very high capacity is assigned to avoid damage from demands
additional_fragility_db.loc[
    'irreparable', [('LS1','Family'),
                    ('LS1','Theta_0'),
                    ('LS1','Theta_1')]] = ['lognormal', 0.02, 0.01]   

# collapse
# capacity is assigned based on the example in the FEMA P58 background documentation
additional_fragility_db.loc[
    'collapse', [('Demand','Directional'),
                    ('Demand','Offset'),
                    ('Demand','Type'), 
                    ('Demand','Unit')]] = [1, 0, 'Peak Interstory Drift Ratio', 'rad']


additional_fragility_db.loc[
    'collapse', [('LS1','Family'),
                 ('LS1','Theta_0'),
                 ('LS1','Theta_1')]] = ['lognormal', 0.05, 0.01]  

# We set the incomplete flag to 0 for the additional components
additional_fragility_db['Incomplete'] = 0

additional_fragility_db.tail(3)

#%%

# load fragility data
PAL.damage.load_damage_model([
    additional_fragility_db,  # This is the extra fragility data we've just created
    'PelicunDefault/fragility_DB_FEMA_P58_2nd.csv' # and this is a table with the default P58 data    
])

#%%

# damage process: see example file

# FEMA P58 uses the following process:
dmg_process = {
    "1_collapse": {
        "DS1": "ALL_NA"
    },
    "2_excessiveRID": {
        "DS1": "irreparable_DS1"
    }
}


#%%

# Now we can run the calculation
PAL.damage.calculate(dmg_process=dmg_process)#, block_batch_size=100) #- for large calculations

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
dmg_plot = (damage_sample.loc[:, component].loc[:,idx[:,:,'2']] / 
            damage_sample.loc[:, component].groupby(level=[0,1], axis=1).sum()).T

px.bar(x=dmg_plot.index.get_level_values(0), y=(dmg_plot>0.5).mean(axis=1), 
       color=dmg_plot.index.get_level_values(1),
       barmode='group',
       labels={
           'x':'Floor',
           'y':'Probability',
           'color': 'Direction'
       },
       title=f'Probability of having more than 50% of component {component} in DS2',
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

# get the consequences used by this assessment
P58_data_for_this_assessment = P58_data.loc[loss_map['BldgRepair'].values[:-2],:]

print(P58_data_for_this_assessment['Incomplete'].sum(), ' components have incomplete consequence models assigned.')

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
additional_consequences.loc[('replacement', 'Cost')] = [0, '1 EA', 'USD_2011', 21600000]
additional_consequences.loc[('replacement', 'Time')] = [0, '1 EA', 'worker_day', 12500]  

additional_consequences

#%%

# Load the loss model to pelicun
PAL.bldg_repair.load_model(
    [additional_consequences,
     "PelicunDefault/bldg_repair_DB_FEMA_P58_2nd.csv"], 
    loss_map)

#%%

# and run the calculations
PAL.bldg_repair.calculate()

#%%

loss_sample = PAL.bldg_repair.sample

print("Size of repair cost & time results: ", sys.getsizeof(loss_sample)/1024/1024, "MB")

loss_sample['COST']['B.20.22.001'].groupby(level=[0,2,3],axis=1).sum().describe([0.1, 0.5, 0.9]).T

#%%
loss_plot = loss_sample.groupby(level=[0, 2], axis=1).sum()['COST'].iloc[:, :-2]

# we add 100 to the loss values to avoid having issues with zeros when creating a log plot
loss_plot += 100

px.box(y=np.tile(loss_plot.columns, loss_plot.shape[0]), 
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
agg_DF_plot = agg_DF.loc[agg_DF['repair_cost'] < 2e7]
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