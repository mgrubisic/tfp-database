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
cost_var = 'cost_mean'
time_var = 'time_u_mean'

# make prediction objects for impacted and non-impacted datasets
df_hit = df[df['impacted'] == 1]
mdl_hit = Prediction(df_hit)
mdl_hit.set_outcome(cost_var)
mdl_hit.test_train_split(0.2)

df_miss = df[df['impacted'] == 0]
mdl_miss = Prediction(df_miss)
mdl_miss.set_outcome(cost_var)
mdl_miss.test_train_split(0.2)

mdl_time_hit = Prediction(df_hit)
mdl_time_hit.set_outcome(time_var)
mdl_time_hit.test_train_split(0.2)

mdl_time_miss = Prediction(df_miss)
mdl_time_miss.set_outcome(time_var)
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
sns.stripplot(x='impacted', y=cost_var, data=df, ax=ax1,
              color='black', jitter=True)
ax1.set_title('Median repair cost', fontsize=subt_font)
ax1.set_ylabel('Cost [USD]', fontsize=axis_font)
ax1.set_xlabel('Impact', fontsize=axis_font)
ax1.set_yscale('log')

sns.boxplot(y="time_u_50%", x= "impacted", data=df,  showfliers=False,
            boxprops={'facecolor': 'none'}, medianprops={'color': 'black'},
            width=0.6, ax=ax2)
sns.stripplot(x='impacted', y=time_var, data=df, ax=ax2,
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

#%% 3d surf

#plt.setp((ax1, ax2, ax3), xticks=np.arange(0.5, 4.0, step=0.5),
#        yticks=np.arange(0.1, 1.1, step=0.1))
#fig=plt.figure(figsize=(13, 4))
#ax1=fig.add_subplot(1, 3, 1)
#ax2=fig.add_subplot(1, 3, 2)
#ax3=fig.add_subplot(1, 3, 3)

# Plot the surface.
#ax1=fig.add_subplot(1, 3, 1, projection='3d')
#surf = ax1.plot_surface(xx, yy, Z, cmap=plt.cm.gist_gray,
#                       linewidth=0, antialiased=False, alpha=0.4)

#ax1.scatter(df[xvar], df[yvar], df[cost_var]/8.1e6, color='white',
#           edgecolors='k', alpha = 0.5)
#
#xlim = ax1.get_xlim()
#ylim = ax1.get_ylim()
#zlim = ax1.get_zlim()
#cset = ax1.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.gist_gray)
#cset = ax1.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.gist_gray)
#cset = ax1.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.gist_gray)
#
#ax1.set_xlabel('Gap ratio', fontsize=axis_font)
#ax1.set_ylabel('$T_M$', fontsize=axis_font)
#ax1.set_zlabel('Mean loss ($)', fontsize=axis_font)
#ax1.set_title('Cost: GPC-SVR', fontsize=subt_font)

#%% Big cost prediction plot (GP-SVR)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plt.close('all')

xvar = 'Tm'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.7, 0.7+res*step-step]
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
plt.setp((ax1, ax2, ax3), yticks=np.arange(0.1, 1.1, step=0.1), ylim=[0.0, 1.0])

yyy = yy[:,1]
cs = ax1.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.7, 1.6, step=0.1))
ax1.clabel(cs, fontsize=label_size)
ax1.set_ylabel('% of replacement cost', fontsize=axis_font)
ax1.set_xlabel('$T_M$', fontsize=axis_font)
ax1.grid()

####################################################################
xvar = 'RI'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.7, 0.7+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax2.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.7, 1.6, step=0.1))

ax2.clabel(cs, fontsize=label_size)
#ax2.set_ylabel('% of replacement', fontsize=axis_font)
ax2.set_xlabel('$R_y$', fontsize=axis_font)
ax2.grid()

####################################################################
xvar = 'zetaM'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.7, 0.7+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)
grid_repair_cost = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_hit.svr,
                                     mdl_miss.svr,
                                     outcome=cost_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_cost)/8.1e6
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax3.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.7, 1.6, step=0.1))
ax3.clabel(cs, fontsize=label_size)

#ax3.set_ylabel('% of replacement', fontsize=axis_font)
ax3.set_xlabel('$\zeta_M$', fontsize=axis_font)
ax3.grid()
plt.show()
fig.tight_layout()

#%% Big downtime prediction plot (GP-KR)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plt.close('all')

xvar = 'Tm'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.7, 0.7+res*step-step]
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_time = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_time_hit.kr,
                                     mdl_time_miss.kr,
                                     outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_time)/4764.71
Z = zz.reshape(xx.shape)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
plt.setp((ax1, ax2, ax3), yticks=np.arange(0.1, 1.1, step=0.1), ylim=[0.0, 1.0])

yyy = yy[:,1]
cs = ax1.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.7, 1.6, step=0.1))
ax1.clabel(cs, fontsize=label_size)
ax1.set_ylabel('% of replacement time', fontsize=axis_font)
ax1.set_xlabel('$T_M$', fontsize=axis_font)
ax1.grid()

####################################################################
xvar = 'RI'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.7, 0.7+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_repair_time = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_time_hit.kr,
                                     mdl_time_miss.kr,
                                     outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_time)/4764.71
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax2.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.7, 1.6, step=0.1))

ax2.clabel(cs, fontsize=label_size)
#ax2.set_ylabel('% of replacement', fontsize=axis_font)
ax2.set_xlabel('$R_y$', fontsize=axis_font)
ax2.grid()

####################################################################
xvar = 'zetaM'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.7, 0.7+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)
grid_repair_time = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_time_hit.kr,
                                     mdl_time_miss.kr,
                                     outcome=time_var)

xx = mdl.xx
yy = mdl.yy
zz = np.array(grid_repair_time)/4764.71
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax3.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.7, 1.6, step=0.1))
ax3.clabel(cs, fontsize=label_size)

#ax3.set_ylabel('% of replacement', fontsize=axis_font)
ax3.set_xlabel('$\zeta_M$', fontsize=axis_font)
ax3.grid()
plt.show()
fig.tight_layout()

#%% Big collapse risk prediction plot (GP-OR)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
axis_font = 18
subt_font = 18
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plt.close('all')

xvar = 'Tm'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.7, 0.7+res*step-step]
X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_drift = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_drift_hit.o_ridge,
                                     mdl_drift_miss.o_ridge,
                                     outcome='max_drift')

from scipy.stats import lognorm
from math import log, exp

xx = mdl.xx
yy = mdl.yy

beta_drift = 0.25
mean_log_drift = exp(log(0.1) - beta_drift*0.9945)
ln_dist = lognorm(s=beta_drift, scale=mean_log_drift)

zz = ln_dist.cdf(np.array(grid_drift))
Z = zz.reshape(xx.shape)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
plt.setp((ax1, ax2, ax3), yticks=np.arange(0.02, 0.22, step=0.02), ylim=[0.0, 0.2])

yyy = yy[:,1]
cs = ax1.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.7, 1.6, step=0.1))
ax1.clabel(cs, fontsize=label_size)
ax1.set_ylabel('Collapse risk', fontsize=axis_font)
ax1.set_xlabel('$T_M$', fontsize=axis_font)
ax1.grid()

####################################################################
xvar = 'RI'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.7, 0.7+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)

grid_drift = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_drift_hit.o_ridge,
                                     mdl_drift_miss.o_ridge,
                                     outcome='max_drift')

xx = mdl.xx
yy = mdl.yy

zz = ln_dist.cdf(np.array(grid_drift))
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax2.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.7, 1.6, step=0.1))

ax2.clabel(cs, fontsize=label_size)
#ax2.set_ylabel('% of replacement', fontsize=axis_font)
ax2.set_xlabel('$R_y$', fontsize=axis_font)
ax2.grid()

####################################################################
xvar = 'zetaM'
yvar = 'gapRatio'

res = 100
step = 0.01
y_bounds = [0.7, 0.7+res*step-step]

X_plot = mdl.make_2D_plotting_space(res, x_var=xvar, y_var=yvar,
                                    y_bounds=y_bounds)
grid_drift = predict_DV(X_plot,
                                     mdl.gpc,
                                     mdl_drift_hit.o_ridge,
                                     mdl_drift_miss.o_ridge,
                                     outcome='max_drift')

xx = mdl.xx
yy = mdl.yy

zz = ln_dist.cdf(np.array(grid_drift))
Z = zz.reshape(xx.shape)

yyy = yy[:,1]
cs = ax3.contour(xx, Z, yy, linewidths=1.1, cmap='copper',
                 levels=np.arange(0.7, 1.6, step=0.1))
ax3.clabel(cs, fontsize=label_size)

#ax3.set_ylabel('% of replacement', fontsize=axis_font)
ax3.set_xlabel('$\zeta_M$', fontsize=axis_font)
ax3.grid()
plt.show()
fig.tight_layout()