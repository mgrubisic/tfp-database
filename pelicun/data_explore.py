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
#%%

import matplotlib.pyplot as plt

plt.close('all')

# df_plot = df[df['cost_90%'] < 8e6]
df_plot = df[df['impacted'] == 1]
fig = plt.figure()
ax = plt.scatter(df_plot['gapRatio'], df_plot['cost_std'])
plt.title('Repair cost scatter')
plt.xlabel('Gap ratio')
plt.ylabel('Repair cost')
plt.yscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['RI'], df_plot['cost_mean'])
plt.title('Repair cost scatter')
plt.xlabel('RI')
plt.ylabel('Repair cost')
plt.yscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['accMax1'], df_plot['cost_mean'])
plt.title('Repair cost scatter')
plt.xlabel('Acceleration')
plt.ylabel('Repair cost')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['gapRatio'], df_plot['accMax1'])
plt.title('Strength and acceleration')
plt.xlabel('gap')
plt.ylabel('Max accel')
plt.yscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['driftMax1'], df_plot['accMax1'])
plt.title('Drift and acceleration')
plt.xlabel('Max drift')
plt.ylabel('Max accel')
plt.yscale('log')
plt.grid(True)
