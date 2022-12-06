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
df["max_drift"] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
df["max_accel"] = df[["accMax0", "accMax1", "accMax2", "accMax3"]].max(axis=1)
df["max_vel"] = df[["velMax0", "velMax1", "velMax2", "velMax3"]].max(axis=1)
#%%

import matplotlib.pyplot as plt

plt.close('all')

# df_plot = df[df['cost_90%'] < 8e6]
df_plot = df[df['impacted'] == 0]
# df_plot = df
fig = plt.figure()
ax = plt.scatter(df_plot['gapRatio'], df_plot['cost_mean'])
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
ax = plt.scatter(df_plot['Tm'], df_plot['max_accel'])
plt.title('Repair cost scatter')
plt.xlabel('Tm')
plt.ylabel('Acceleration')
plt.yscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['zetaM'], df_plot['max_accel'])
plt.title('Repair cost scatter')
plt.xlabel('Damping')
plt.ylabel('Acceleration')
plt.yscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['max_accel'], df_plot['cost_mean'])
plt.title('Acceleration and repair cost')
plt.xlabel('Acceleration (g)')
plt.ylabel('Repair cost')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)

fig = plt.figure()
ax = plt.scatter(df_plot['max_drift'], df_plot['cost_mean'])
plt.title('Drift and repair cost')
plt.xlabel('Max drift (PID%)')
plt.ylabel('Repair cost')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
