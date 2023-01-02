############################################################################
#               ML plotter

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  plots for IALCCE paper

# Open issues:  

############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prediction_model import Prediction, predict_DV
plt.close('all')
idx = pd.IndexSlice
pd.options.display.max_rows = 30

#%% concat with other data
loss_data = pd.read_csv('./results/loss_estimate_data.csv', index_col=None)
full_isolation_data = pd.read_csv('full_isolation_data.csv', index_col=None)

df = pd.concat([full_isolation_data, loss_data], axis=1)
df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)

#%% Prepare data

# make prediction objects for impacted and non-impacted datasets
df_hit = df[df['impacted'] == 1]
mdl_hit = Prediction(df_hit)
mdl_hit.set_outcome('cost_50%')
mdl_hit.test_train_split(0.2)

df_miss = df[df['impacted'] == 0]
mdl_miss = Prediction(df_miss)
mdl_miss.set_outcome('cost_50%')
mdl_miss.test_train_split(0.2)

mdl_time_hit = Prediction(df_hit)
mdl_time_hit.set_outcome('time_u_50%')
mdl_time_hit.test_train_split(0.2)

mdl_time_miss = Prediction(df_miss)
mdl_time_miss.set_outcome('time_u_50%')
mdl_time_miss.test_train_split(0.2)

mdl_drift_hit = Prediction(df_hit)
mdl_drift_hit.set_outcome('max_drift')
mdl_drift_hit.test_train_split(0.2)

mdl_drift_miss = Prediction(df_miss)
mdl_drift_miss.set_outcome('max_drift')
mdl_drift_miss.test_train_split(0.2)
#%% fit impact (gp classification)

# prepare the problem
mdl = Prediction(df)
mdl.set_outcome('impacted')
mdl.test_train_split(0.2)

mdl.fit_gpc(kernel_name='rbf_iso')

# predict the entire dataset
preds_imp = mdl.gpc.predict(mdl.X)
probs_imp = mdl.gpc.predict_proba(mdl.X)

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl.y, preds_imp).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

#%% Classification plot

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
import matplotlib as mpl
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plt.close('all')
# make grid and plot classification predictions

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

xvar = 'gapRatio'
yvar = 'RI'
X_plot = mdl.make_2D_plotting_space(100, x_var=xvar, y_var=yvar)
xx = mdl.xx
yy = mdl.yy
Z = mdl.gpc.predict_proba(mdl.X_plot)[:, 1]
Z = Z.reshape(xx.shape)

#ax1.imshow(
#        Z,
#        interpolation="nearest",
#        extent=(xx.min(), xx.max(),
#                yy.min(), yy.max()),
#        aspect="auto",
#        origin="lower",
#        cmap=plt.cm.Greys,
#    )

plt_density = 100
cs = ax1.contour(xx, yy, Z, linewidths=1.1, cmap='copper',
                 levels=np.linspace(0.1,1.0,num=10))
ax1.clabel(cs, fontsize=label_size)

#ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2,
#            linestyles="dashed", colors='black')

ax1.scatter(mdl.X_train[xvar][:plt_density],
            mdl.X_train[yvar][:plt_density],
            s=30, c=mdl.y_train[:plt_density],
            cmap=plt.cm.copper, edgecolors="k")
ax1.set_title(r'$T_M = 3.24$ s, $\zeta_M = 0.155$', fontsize=subt_font)
ax1.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax1.set_ylabel(r'$R_y$', fontsize=axis_font)

####################################################################
xvar = 'gapRatio'
yvar = 'Tm'
X_plot = mdl.make_2D_plotting_space(100, x_var=xvar, y_var=yvar)
xx = mdl.xx
yy = mdl.yy
Z = mdl.gpc.predict_proba(mdl.X_plot)[:, 1]
Z = Z.reshape(xx.shape)

#ax1.imshow(
#        Z,
#        interpolation="nearest",
#        extent=(xx.min(), xx.max(),
#                yy.min(), yy.max()),
#        aspect="auto",
#        origin="lower",
#        cmap=plt.cm.Greys,
#    )

plt_density = 100
cs = ax2.contour(xx, yy, Z, linewidths=1.1, cmap='copper',
                 levels=np.linspace(0.1,1.0,num=10))
ax2.clabel(cs, fontsize=label_size)

#ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2,
#            linestyles="dashed", colors='black')

ax2.scatter(mdl.X_train[xvar][:plt_density],
            mdl.X_train[yvar][:plt_density],
            s=30, c=mdl.y_train[:plt_density],
            cmap=plt.cm.copper, edgecolors="k")
ax2.set_title(r'$R_y= 1.22$ s, $\zeta_M = 0.155$', fontsize=subt_font)
ax2.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax2.set_ylabel(r'$T_M$', fontsize=axis_font)

####################################################################
xvar = 'gapRatio'
yvar = 'zetaM'
X_plot = mdl.make_2D_plotting_space(100, x_var=xvar, y_var=yvar)
xx = mdl.xx
yy = mdl.yy
Z = mdl.gpc.predict_proba(mdl.X_plot)[:, 1]
Z = Z.reshape(xx.shape)

#ax1.imshow(
#        Z,
#        interpolation="nearest",
#        extent=(xx.min(), xx.max(),
#                yy.min(), yy.max()),
#        aspect="auto",
#        origin="lower",
#        cmap=plt.cm.Greys,
#    )

plt_density = 100
cs = ax3.contour(xx, yy, Z, linewidths=1.1, cmap='copper',
                 levels=np.linspace(0.1,1.0,num=10))
ax3.clabel(cs, fontsize=label_size)

#ax1.contour(xx, yy, Z, levels=[0.5], linewidths=2,
#            linestyles="dashed", colors='black')

sc = ax3.scatter(mdl.X_train[xvar][:plt_density],
            mdl.X_train[yvar][:plt_density],
            s=30, c=mdl.y_train[:plt_density],
            cmap=plt.cm.copper, edgecolors="k")
ax3.set_title(r'$R_y= 1.22$ s, $T_M = 3.24$ s', fontsize=subt_font)
ax3.set_xlabel(r'Gap ratio', fontsize=axis_font)
ax3.set_ylabel(r'$\zeta_M$', fontsize=axis_font)
lg = ax3.legend(*sc.legend_elements(), loc="lower right", title="Impact",
           fontsize=subt_font)
lg.get_title().set_fontsize(axis_font) #legend 'Title' fontsize

fig.tight_layout()
plt.show()

#%%

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plt.close('all')
import seaborn as sns

# make grid and plot classification predictions

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
sns.boxplot(y="cost_50%", x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, medianprops={'color': 'black'},
            width=0.6, ax=ax1)
sns.stripplot(x='impacted', y='cost_50%', data=df, ax=ax1,
              color='black', jitter=True)
ax1.set_title('Median repair cost', fontsize=subt_font)
ax1.set_ylabel('Cost [USD]', fontsize=axis_font)
ax1.set_xlabel('Impact', fontsize=axis_font)
ax1.set_yscale('log')

sns.boxplot(y="time_u_50%", x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, medianprops={'color': 'black'},
            width=0.6, ax=ax2)
sns.stripplot(x='impacted', y='time_u_50%', data=df, ax=ax2,
              color='black', jitter=True)
ax2.set_title('Median sequential repair time', fontsize=subt_font)
ax2.set_ylabel('Time [worker-day]', fontsize=axis_font)
ax2.set_xlabel('Impact', fontsize=axis_font)
ax2.set_yscale('log')

sns.boxplot(y="replacement_freq", x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, medianprops={'color': 'black'},
            width=0.5, ax=ax3)
sns.stripplot(x='impacted', y='replacement_freq', data=df, ax=ax3,
              color='black', jitter=True)
ax3.set_title('Replacement frequency', fontsize=subt_font)
ax3.set_ylabel('Replacement frequency', fontsize=axis_font)
ax3.set_xlabel('Impact', fontsize=axis_font)
fig.tight_layout()

#%% regression models

# Fit costs (SVR)

# fit impacted set
mdl_hit.fit_svr()
mdl_time_hit.fit_kernel_ridge(kernel_name='rbf')
mdl_drift_hit.fit_ols_ridge()

# fit no impact set
mdl_miss.fit_svr()
mdl_time_miss.fit_kernel_ridge(kernel_name='rbf')
mdl_drift_miss.fit_ols_ridge()
