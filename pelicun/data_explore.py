############################################################################
#               Loss and damage data

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  Visualize loss data

# Open issues:  (1) 

############################################################################

import pandas as pd
idx = pd.IndexSlice
pd.options.display.max_rows = 30


#%% concat with other data
loss_data = pd.read_csv('./results/loss_estimate_data.csv', index_col=None)
full_isolation_data = pd.read_csv('full_isolation_data.csv', index_col=None)

df = pd.concat([full_isolation_data, loss_data], axis=1)
df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df['max_accel'] = df[["accMax0", "accMax1", "accMax2", "accMax3"]].max(axis=1)
df['max_vel'] = df[["velMax0", "velMax1", "velMax2", "velMax3"]].max(axis=1)
#%%

import matplotlib.pyplot as plt

plt.close('all')

# df_plot = df[df['cost_90%'] < 8e6]
# df_plot = df[df['impacted'] == 0]
# df_plot = df[df['replacement_freq'] < 0.3]
df_plot = df
fig = plt.figure()
ax = plt.scatter(df_plot['gapRatio'], df_plot['cost_50%'], alpha=0.5)
plt.title('Repair cost no impact')
plt.xlabel('Gap ratio')
plt.ylabel('Median repair cost ($)')
plt.yscale('log')

plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['RI'], df_plot['cost_50%'], alpha=0.5)
plt.title('Repair cost no impact')
plt.xlabel('RI')
plt.ylabel('Median repair cost ($)')
plt.yscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['max_accel'], df_plot['cost_50%'], alpha=0.5)
plt.title('Acceleration and repair cost | no impact')
plt.xlabel('Max floor acceleration (g)')
plt.ylabel('Median repair cost ($)')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['max_drift'], df_plot['cost_50%'], alpha=0.5)
plt.title('Drift and repair cost | no impact')
plt.xlabel('Max drift [PID%]')
plt.ylabel('Median repair cost ($)')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['max_drift'], df_plot['max_accel'], alpha=0.5)
plt.title('Drift and accel | no impact')
plt.xlabel('Max drift [PID%]')
plt.ylabel('Max floor acceleration (g)')
plt.yscale('log')
plt.grid(True)

#%%
df_plot = df

plt.close('all')

fig = plt.figure()
ax = plt.scatter(df_plot['max_accel'], df_plot['collapse_freq'], alpha=0.5)
plt.title('Acceleration and collapse frequency')
plt.xlabel('Max floor acceleration (g)')
plt.ylabel('MC collapse frequency')
plt.xscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['max_drift'], df_plot['collapse_freq'], alpha=0.5)
plt.title('Drift and collapse frequency')
plt.xlabel('Max drift [PID%]')
plt.ylabel('MC collapse frequency')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['gapRatio'], df_plot['replacement_freq'], alpha=0.5)
plt.title('Gap and collapse frequency')
plt.xlabel('Gap ratio')
plt.ylabel('MC replacement frequency')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['RI'], df_plot['replacement_freq'], alpha=0.5)
plt.title('Structure strength and collapse frequency')
plt.xlabel('RI')
plt.ylabel('MC replacement frequency')
plt.grid(True)

#%%
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

df_plot = df[df['impacted'] == 0]
fig = px.scatter_3d(df_plot, x='max_drift', y='max_accel', z='cost_50%',
              color='time_u_50%', opacity=0.5, log_z=True, log_x=True, log_y=True)
fig.show()

df_plot = df
fig = px.scatter_3d(df_plot, x='gapRatio', y='RI', z='cost_50%',
              color='max_accel', log_z=True)
fig.show()

df_plot = df[df['impacted'] == 0]
fig = px.scatter(df_plot, x='max_drift', y='max_accel',
                 opacity=0.5,
              log_x=True, log_y=True,
              marginal_x ='histogram', marginal_y='histogram')
fig.show()
