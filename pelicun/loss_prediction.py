############################################################################
#               ML prediction models for isolator loss data

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  ML models

# Open issues:  (1) 

############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prediction_model import Prediction
plt.close('all')
idx = pd.IndexSlice
pd.options.display.max_rows = 30

## temporary spyder debugger error hack
#import collections
#collections.Callable = collections.abc.Callable

#%% concat with other data
loss_data = pd.read_csv('./results/loss_estimate_data.csv', index_col=None)
full_isolation_data = pd.read_csv('full_isolation_data.csv', index_col=None)

df = pd.concat([full_isolation_data, loss_data], axis=1)
# df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
# df['max_accel'] = df[["accMax0", "accMax1", "accMax2", "accMax3"]].max(axis=1)
# df['max_vel'] = df[["velMax0", "velMax1", "velMax2", "velMax3"]].max(axis=1)
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

# prepare the problem
mdl = Prediction(df)
mdl.set_outcome('impacted')
mdl.test_train_split(0.2)

#%% fit impact (SVC)
# fit SVM classification for impact
# lower neg_wt = penalize false negatives more

mdl.fit_svc(neg_wt=1.0, kernel_name='sigmoid')
#mdl.fit_svc(neg_wt=0.85, kernel_name='rbf')

# predict the entire dataset
preds_imp = mdl.svc.predict(mdl.X)

# note: SVC probabilities are NOT well calibrated
probs_imp = mdl.svc.predict_proba(mdl.X)

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl.y, preds_imp).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

# make grid and plot classification predictions
X_plot = mdl.make_2D_plotting_space(100)
mdl.plot_classification(mdl.svc)

#%% fit impact (logistic classification)

# fit logistic classification for impact
mdl.fit_log_reg(neg_wt=0.85)

# predict the entire dataset
preds_imp = mdl.log_reg.predict(mdl.X)
probs_imp = mdl.log_reg.predict_proba(mdl.X)

# we've done manual CV to pick the hyperparams that trades some accuracy
# in order to lower false negatives
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(mdl.y, preds_imp).ravel()
print('False negatives: ', fn)
print('False positives: ', fp)

# make grid and plot classification predictions
X_plot = mdl.make_2D_plotting_space(100)
mdl.plot_classification(mdl.log_reg)

#%% fit impact (gp classification)

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

# make grid and plot classification predictions
X_plot = mdl.make_2D_plotting_space(100)
mdl.plot_classification(mdl.gpc)
#%% Fit costs (SVR)

# fit impacted set
mdl_hit.fit_svr()
cost_pred_hit = mdl_hit.svr.predict(mdl_hit.X_test)
comparison_cost_hit = np.array([mdl_hit.y_test, 
                                      cost_pred_hit]).transpose()
        
# fit no impact set
mdl_miss.fit_svr()
cost_pred_miss = mdl_miss.svr.predict(mdl_miss.X_test)
comparison_cost_miss = np.array([mdl_miss.y_test, 
                                      cost_pred_miss]).transpose()

mdl_miss.make_2D_plotting_space(100)

xx = mdl_miss.xx
yy = mdl_miss.yy
Z = mdl_miss.svr.predict(mdl_miss.X_plot)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(df_miss['gapRatio'], df_miss['RI'], df_miss['cost_50%'],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Median loss ($)')
ax.set_title('Median cost predictions given no impact (SVR)')
ax.set_zlim([0, 1e5])
plt.show()

#%% Fit costs (kernel ridge)

kernel_type = 'polynomial'

# fit impacted set
mdl_hit.fit_kernel_ridge(kernel_name=kernel_type)
cost_pred_hit = mdl_hit.kr.predict(mdl_hit.X_test)
comparison_cost_hit = np.array([mdl_hit.y_test, 
                                      cost_pred_hit]).transpose()
        
# fit no impact set
mdl_miss.fit_kernel_ridge(kernel_name=kernel_type)
cost_pred_miss = mdl_miss.kr.predict(mdl_miss.X_test)
comparison_cost_miss = np.array([mdl_miss.y_test, 
                                      cost_pred_miss]).transpose()

mdl_miss.make_2D_plotting_space(100)

xx = mdl_miss.xx
yy = mdl_miss.yy
Z = mdl_miss.kr.predict(mdl_miss.X_plot)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(df_miss['gapRatio'], df_miss['RI'], df_miss['cost_50%'],
           edgecolors='k')

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Median loss ($)')
ax.set_title('Median cost predictions given no impact (Polynomial kernel ridge)')
ax.set_zlim([0, 1e6])
plt.show()

#%% Fit costs (GP regression)

#kernel_type = 'rbf_iso'
#
## fit impacted set
#mdl_hit.fit_gpr(kernel_name=kernel_type)
#cost_pred_hit = mdl_hit.gpr.predict(mdl_hit.X_test)
#comparison_cost_hit = np.array([mdl_hit.y_test, 
#                                      cost_pred_hit]).transpose()
#        
## fit no impact set
#mdl_miss.fit_gpr(kernel_name=kernel_type)
#cost_pred_miss = mdl_miss.gpr.predict(mdl_miss.X_test)
#comparison_cost_miss = np.array([mdl_miss.y_test, 
#                                      cost_pred_miss]).transpose()
#
#mdl_miss.make_2D_plotting_space(100)
#
#xx = mdl_miss.xx
#yy = mdl_miss.yy
#Z = mdl_miss.gpr.predict(mdl_miss.X_plot)
#Z = Z.reshape(xx.shape)
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
## Plot the surface.
#surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#ax.scatter(df_miss['gapRatio'], df_miss['RI'], df_miss['cost_50%'],
#           edgecolors='k')
#
#ax.set_xlabel('Gap ratio')
#ax.set_ylabel('Ry')
#ax.set_zlabel('Median loss ($)')
#ax.set_title('Median cost predictions given no impact (GP regression)')
#ax.set_zlim([0, 1e6])
#plt.show()

#%% Fit costs (regular ridge)

# sensitive to alpha. keep alpha > 1e-2 since smaller alphas will result in 
# flat line to minimize the outliers' error

# results in a fit similar to kernel ridge

# fit impacted set
mdl_hit.fit_ols_ridge()
cost_pred_hit = mdl_hit.o_ridge.predict(mdl_hit.X_test)
comparison_cost_hit = np.array([mdl_hit.y_test, 
                                      cost_pred_hit]).transpose()
        
# fit no impact set
mdl_miss.fit_ols_ridge()
cost_pred_miss = mdl_miss.o_ridge.predict(mdl_miss.X_test)
comparison_cost_miss = np.array([mdl_miss.y_test, 
                                      cost_pred_miss]).transpose()

#%% aggregate the two models
mdl.predict_loss(mdl.log_reg, mdl_hit.svr, mdl_miss.svr)
comparison_cost = np.array([df['cost_50%'],
                            np.ravel(mdl.median_loss_pred)]).transpose()

#%% Big cost prediction plot (SVC-SVR)

X_plot = mdl.make_2D_plotting_space(100)

grid_mdl = Prediction(X_plot)
grid_mdl.predict_loss(mdl.svc, mdl_hit.svr, mdl_miss.svr)
grid_mdl.make_2D_plotting_space(100)

xx = grid_mdl.xx
yy = grid_mdl.yy
Z = np.array(grid_mdl.median_loss_pred)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df['cost_50%'],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Median loss ($)')
ax.set_title('Median cost predictions: SVC-impact, SVR-loss')
plt.show()

#%% Big cost prediction plot (LR-SVR)

grid_mdl.predict_loss(mdl.log_reg, mdl_hit.svr, mdl_miss.svr)
grid_mdl.make_2D_plotting_space(100)

xx = grid_mdl.xx
yy = grid_mdl.yy
Z = np.array(grid_mdl.median_loss_pred)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df['cost_50%'],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Median loss ($)')
ax.set_title('Median cost predictions: LR-impact, SVR-loss')
plt.show()

#%% Big cost prediction plot (GP-SVR)

grid_mdl.predict_loss(mdl.gpc, mdl_hit.svr, mdl_miss.svr)
grid_mdl.make_2D_plotting_space(100)

xx = grid_mdl.xx
yy = grid_mdl.yy
Z = np.array(grid_mdl.median_loss_pred)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df['cost_50%'],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Median loss ($)')
ax.set_title('Median cost predictions: GP-impact, SVR-loss')
plt.show()

#%% Big cost prediction plot (GP-KR)

grid_mdl.predict_loss(mdl.gpc, mdl_hit.kr, mdl_miss.kr)
grid_mdl.make_2D_plotting_space(100)

xx = grid_mdl.xx
yy = grid_mdl.yy
Z = np.array(grid_mdl.median_loss_pred)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df['cost_50%'],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Median loss ($)')
ax.set_title('Median cost predictions: GP-impact, KR-loss')
plt.show()

#%% Fit downtime (SVR)
# make prediction objects for impacted and non-impacted datasets
mdl_time_hit = Prediction(df_hit)
mdl_time_hit.set_outcome('time_u_50%')
mdl_time_hit.test_train_split(0.2)

mdl_time_miss = Prediction(df_miss)
mdl_time_miss.set_outcome('time_u_50%')
mdl_time_miss.test_train_split(0.2)

# fit impacted set
mdl_time_hit.fit_svr()
time_pred_hit = mdl_time_hit.svr.predict(mdl_time_hit.X_test)
comparison_time_hit = np.array([mdl_time_hit.y_test, 
                                      time_pred_hit]).transpose()
        
# fit no impact set
mdl_time_miss.fit_svr()
time_pred_miss = mdl_time_miss.svr.predict(mdl_time_miss.X_test)
comparison_time_miss = np.array([mdl_time_miss.y_test, 
                                      time_pred_miss]).transpose()

#mdl_time_miss.make_2D_plotting_space(100)
#
#xx = mdl_time_miss.xx
#yy = mdl_time_miss.yy
#Z = mdl_time_miss.svr.predict(mdl_time_miss.X_plot)
#Z = Z.reshape(xx.shape)
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
## Plot the surface.
#surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#ax.scatter(df_miss['gapRatio'], df_miss['RI'], df_miss['time_u_50%'],
#           edgecolors='k')
#
#ax.set_xlabel('Gap ratio')
#ax.set_ylabel('Ry')
#ax.set_zlabel('Median downtime (worker-day)')
#ax.set_title('Median sequential downtime predictions given no impact (SVR)')
#ax.set_zlim([0, 500])
#plt.show()

#%% Big downtime prediction plot (GP-SVR)

grid_mdl.predict_loss(mdl.gpc, mdl_time_hit.svr, mdl_time_miss.svr)
grid_mdl.make_2D_plotting_space(100)

xx = grid_mdl.xx
yy = grid_mdl.yy
Z = np.array(grid_mdl.median_loss_pred)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)

ax.scatter(df['gapRatio'], df['RI'], df['time_u_50%'],
           edgecolors='k', alpha = 0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
cset = ax.contour(xx, yy, Z, zdir='z', offset=zlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='x', offset=xlim[0], cmap=plt.cm.coolwarm)
cset = ax.contour(xx, yy, Z, zdir='y', offset=ylim[1], cmap=plt.cm.coolwarm)

ax.set_xlabel('Gap ratio')
ax.set_ylabel('Ry')
ax.set_zlabel('Median downtime (worker-days)')
ax.set_title('Median sequential downtime predictions: GP-impact, SVR-loss')
plt.show()

#%% Testing the design space
import time

res_des = 30
X_space = mdl.make_design_space(res_des)

des_mdl = Prediction(X_space)
t0 = time.time()
des_mdl.predict_loss(mdl.log_reg, mdl_hit.svr, mdl_miss.svr)
tp = time.time() - t0
print("LR-SVR prediction for %d inputs in %.3f s" % (X_space.shape[0], tp))

#%% Fit costs (SVR, across all data)

## fit impacted set
#mdl = Prediction(df)
#mdl.set_outcome('cost_50%')
#mdl.test_train_split(0.2)
#mdl.fit_svr()
#
#X_plot = mdl.make_2D_plotting_space(100)
#
#xx = mdl.xx
#yy = mdl.yy
#Z = mdl.svr.predict(mdl.X_plot)
#Z = Z.reshape(xx.shape)
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
## Plot the surface.
#surf = ax.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#ax.scatter(df['gapRatio'], df['RI'], df['cost_50%'],
#           edgecolors='k')
#
#ax.set_xlabel('Gap ratio')
#ax.set_ylabel('Ry')
#ax.set_zlabel('Median loss ($)')
#ax.set_title('Median cost predictions across all data (SVR)')
#plt.show()
        

#%% dirty test prediction plots for cost

##plt.close('all')
#plt.figure()
#plt.scatter(mdl_miss.X_test['RI'], mdl_miss.y_test)
#plt.scatter(mdl_miss.X_test['RI'], cost_pred_miss)
#
#plt.figure()
#plt.scatter(mdl_hit.X_test['RI'], mdl_hit.y_test)
#plt.scatter(mdl_hit.X_test['RI'], cost_pred_hit)

#%% Other plotting

# idea for plots

# stats-style qq error plot
# slice the 3D predictions into contours and label important values
# Tm, zeta plots