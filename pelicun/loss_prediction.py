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
#%% new class
class Prediction:
    
    # sets up the problem by grabbing the x covariates
    def __init__(self, data):
        self._raw_data = data
        self.k = len(data)
        self.X = data[['gapRatio', 'RI', 'zetaM', 'Tm']]
        
    # sets up prediction variable
    def set_outcome(self, outcome_var):
        self.y = self._raw_data[[outcome_var]]
        
    # if classification is done, plot the predictions
    def plot_classification(self, mdl_clf):
        import matplotlib.pyplot as plt
        
        xx = self.xx
        yy = self.yy
        if 'gpc' in mdl_clf.named_steps.keys():
            Z = mdl_clf.predict_proba(self.X_plot)[:, 1]
        else:
            Z = mdl_clf.decision_function(self.X_plot)
            
        Z = Z.reshape(xx.shape)
        
        plt.figure()
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(),
                    yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
            cmap=plt.cm.PuOr_r,
        )
            
        plt_density = 100
        
        if 'gpc' in mdl_clf.named_steps.keys():
            plt.contour(xx, yy, Z, levels=[0.5],
                        linewidths=2, linestyles="dashed")
        else:
            plt.contour(xx, yy, Z, levels=[0],
                        linewidths=2, linestyles="dashed")
        plt.scatter(self.X_train['gapRatio'][:plt_density],
                    self.X_train['RI'][:plt_density],
                    s=30, c=self.y_train[:plt_density],
                    cmap=plt.cm.Paired, edgecolors="k")
        plt.xlabel('Gap ratio')
        plt.ylabel('Ry')
        if 'svc' in mdl_clf.named_steps.keys():
            plt.title('Impact classification (SVC)')
        elif 'log_reg' in mdl_clf.named_steps.keys():
            plt.title('Impact classification (logistic)')
        elif 'gpc' in mdl_clf.named_steps.keys():
            plt.title('Impact classification (GP)')
        plt.show()
        
    # make a generalized 2D plotting grid, defaulted to gap and Ry
    # grid is based on the bounds of input data
    def make_2D_plotting_space(self, res, x_var='gapRatio', y_var='RI'):
        xx, yy = np.meshgrid(np.linspace(min(self.X[x_var]),
                                         max(self.X[x_var]),
                                         res),
                             np.linspace(min(self.X[y_var]),
                                         max(self.X[y_var]),
                                         res))

        if (x_var=='gapRatio') and (y_var=='RI'):
            third_var = 'zetaM'
            fourth_var = 'Tm'
           
        self.xx = xx
        self.yy = yy
        self.X_plot = pd.DataFrame({x_var:xx.ravel(),
                             y_var:yy.ravel(),
                             third_var:np.repeat(self.X[third_var].median(),
                                                 res*res),
                             fourth_var:np.repeat(self.X[fourth_var].median(), 
                                                  res*res)})
                             
        return(self.X_plot)
        
    # train test split to be done before any learning 
    def test_train_split(self, percentage):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                            test_size=percentage,
                                                            random_state=985)
        
        from numpy import ravel
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = ravel(y_train)
        self.y_test = ravel(y_test)
 
###############################################################################
    # Classification models
###############################################################################       
    
    # Train GP classifier
    def fit_gpc(self, kernel_name):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessClassifier
        import sklearn.gaussian_process.kernels as krn
        
        if kernel_name=='rbf_ard':
            kernel = 1.0 * krn.RBF([1.0, 1.0, 1.0, 1.0])
        elif kernel_name=='rbf_iso':
            kernel = 1.0 * krn.RBF(1.0)
        elif kernel_name=='rq':
            kernel = 0.5**2 * krn.RationalQuadratic(length_scale=1.0,
                                                    alpha=1.0)
        elif kernel_name == 'matern_iso':
            kernel = 1.0 * krn.Matern(
                    length_scale=1.0, 
                    nu=1.5)
        elif kernel_name == 'matern_ard':
            kernel = 1.0 * krn.Matern(
                    length_scale=[1.0, 1.0, 1.0, 1.0], 
                    nu=1.5)

        kernel = kernel + krn.WhiteKernel(noise_level=0.5)
        # pipeline to scale -> GPC
        gp_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('gpc', GaussianProcessClassifier(kernel=kernel,
                                                  warm_start=True,
                                                  random_state=985,
                                                  max_iter_predict=250))
                ])
    
        gp_pipe.fit(self.X_train, self.y_train)
        tr_scr = gp_pipe.score(self.X_train, self.y_train)
        print("The GP training score is %0.2f"
              %tr_scr)
        
        te_scr = gp_pipe.score(self.X_test, self.y_test)
        print('GP testing score: %0.2f' %te_scr)
        self.gpc = gp_pipe
    
    # Train SVM regression
    def fit_svr(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.model_selection import GridSearchCV
        
        # pipeline to scale -> SVR
        sv_pipe = Pipeline([('scaler', StandardScaler()),
                            ('svr', SVR(kernel='rbf'))])
        
        # cross-validate several parameters
        parameters = [
            {'svr__C':[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
             'svr__epsilon':[0.01, 0.1, 1.0],
             'svr__gamma':np.logspace(-2, 2, 3)}
            ]
        
        svr_cv = GridSearchCV(sv_pipe, param_grid=parameters)
        svr_cv.fit(self.X_train, self.y_train)
        
        print("The best SVR parameters are %s"
              % (svr_cv.best_params_))
        
        # set pipeline to use CV params
        sv_pipe.set_params(**svr_cv.best_params_)
        sv_pipe.fit(self.X_train, self.y_train)
        
        self.svr = sv_pipe
        
    # Train logistic classification
    # TODO: kernelize
    def fit_log_reg(self, neg_wt=1.0):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegressionCV
        
        # pipeline to scale -> logistic
        wts = {0: neg_wt, 1:1.0}
        log_reg_pipe = Pipeline([('scaler', StandardScaler()),
                                 ('log_reg', LogisticRegressionCV(
                                         class_weight=wts))])
        
        # LRCV finds optimum C value, L2 penalty
        log_reg_pipe.fit(self.X_train, self.y_train)
        
        # Get test accuracy
        C = log_reg_pipe._final_estimator.C_
        tr_scr = log_reg_pipe.score(self.X_train, self.y_train)
        
        print('The best logistic C value is %f with a training score of %0.2f'
              % (C, tr_scr))
        
        te_scr = log_reg_pipe.score(self.X_test, self.y_test)
        print('Logistic testing score: %0.2f'
              %te_scr)
        
        self.log_reg = log_reg_pipe
        
    # Train SVM classification
    def fit_svc(self, neg_wt=1.0):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        
        # pipeline to scale -> SVC
        wts = {0: neg_wt, 1:1.0}
        sv_pipe = Pipeline([('scaler', StandardScaler()),
                            ('svc', SVC(kernel='rbf', gamma='auto',
                                        probability=True,
                                        class_weight=wts))])
        
        # cross-validate several parameters
        parameters = [
            {'svc__C':[0.1, 1.0, 10.0, 100.0, 1000.0]}
            ]
        
        svc_cv = GridSearchCV(sv_pipe, param_grid=parameters)
        svc_cv.fit(self.X_train, self.y_train)
        
        # set pipeline to use CV params
        sv_pipe.set_params(**svc_cv.best_params_)
        sv_pipe.fit(self.X_train, self.y_train)
        
        print("The best SVC parameters are %s with a training score of %0.2f"
              % (svc_cv.best_params_, svc_cv.best_score_))
        
        te_scr = sv_pipe.score(self.X_test, self.y_test)
        print('SVC testing score: %0.2f' %te_scr)
        self.svc = sv_pipe
        
    # TODO: if params passed, don't do CV

###############################################################################
    # Regression models
###############################################################################
    # Train kernel ridge regression
    def fit_kernel_ridge(self, kernel_name='rbf'):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV
        
        kr_pipe = Pipeline([('scaler', StandardScaler()),
                             ('kr', KernelRidge(kernel=kernel_name))])
        
        # cross-validate several parameters
        parameters = [
            {'kr__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
             'kr__gamma':np.logspace(-2, 2, 5)}
            ]
        
        kr_cv = GridSearchCV(kr_pipe, param_grid=parameters)
        kr_cv.fit(self.X_train, self.y_train)
        
        # set pipeline to use CV params
        print("The best kernel ridge parameters are %s"
              % (kr_cv.best_params_))
        kr_pipe.set_params(**kr_cv.best_params_)
        kr_pipe.fit(self.X_train, self.y_train)
        
        self.kr = kr_pipe
        
    # Train GP regression
    def fit_gpr(self, kernel_name):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.gaussian_process import GaussianProcessRegressor
        import sklearn.gaussian_process.kernels as krn
        
        if kernel_name=='rbf_ard':
            kernel = 1.0 * krn.RBF([1.0, 1.0, 1.0, 1.0])
        elif kernel_name=='rbf_iso':
            kernel = 1.0 * krn.RBF(1.0)
        elif kernel_name=='rq':
            kernel = 0.5**2 * krn.RationalQuadratic(length_scale=1.0,
                                                    alpha=1.0)
        elif kernel_name == 'matern_iso':
            kernel = 1.0 * krn.Matern(
                    length_scale=1.0, 
                    nu=1.5)
        elif kernel_name == 'matern_ard':
            kernel = 1.0 * krn.Matern(
                    length_scale=[1.0, 1.0, 1.0, 1.0], 
                    nu=1.5)
            
        # pipeline to scale -> GPR
        gp_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('gpr', GaussianProcessRegressor(kernel=kernel,
                                                 random_state=985,
                                                 n_restarts_optimizer=10))
                ])
    
        gp_pipe.fit(self.X_train, self.y_train)
        
        self.gpr = gp_pipe
        
    # Train regular ridge regression
    def fit_ols_ridge(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        
        or_pipe = Pipeline([('scaler', StandardScaler()),
                             ('o_ridge', RidgeCV(alphas=np.logspace(-2, 2, 5)))]
            )
        
        or_pipe.fit(self.X_train, self.y_train)
        
        self.o_ridge = or_pipe
        
###############################################################################
    # Full prediction models
###############################################################################
    
    # assumes that problem is already created
    def predict_loss(self, impact_pred_mdl, hit_loss_mdl, miss_loss_mdl):
        
        # two ways of doing this
        # 1) predict impact first (binary)), then fit the impact predictions 
        # with the impact-only SVR and likewise with non-impacts. This creates
        # two tiers of predictions that are relatively flat (impact dominated)
        # 2) using expectations, get probabilities of collapse and weigh the
        # two (cost|impact) regressions with Pr(impact). Creates smooth
        # predictions that are somewhat moderate
        
#        # get points that are predicted impact from full dataset
#        preds_imp = impact_pred_mdl.svc.predict(self.X)
#        df_imp = self.X[preds_imp == 1]
        
        # get probability of impact
        probs_imp = impact_pred_mdl.predict_proba(self.X)
        
        miss_prob = probs_imp[:,0]
        hit_prob = probs_imp[:,1]
        
        # weight with probability of collapse
        # E[Loss] = (impact loss)*Pr(impact) + (no impact loss)*Pr(no impact)
        # run SVR_hit model on this dataset
        loss_pred_hit = pd.DataFrame(
                {'median_loss_pred':np.multiply(
                        hit_loss_mdl.predict(self.X),
                        hit_prob)},
                    index=self.X.index)
                
        
#        # get points that are predicted no impact from full dataset
#        df_mss = self.X[preds_imp == 0]
        
        # run SVR_miss model on this dataset
        loss_pred_miss = pd.DataFrame(
                {'median_loss_pred':np.multiply(
                        miss_loss_mdl.predict(self.X),
                        miss_prob)},
                    index=self.X.index)
        
        self.median_loss_pred = loss_pred_hit+loss_pred_miss
            
        # self.median_loss_pred = pd.concat([loss_pred_hit,loss_pred_miss], 
        #                                   axis=0).sort_index(ascending=True)
        

 
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
mdl.fit_svc(neg_wt=0.6)

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
        
scr = mdl_miss.svr.score(mdl_miss.X_test, mdl_miss.y_test)

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
        
scr = mdl_miss.kr.score(mdl_miss.X_test, mdl_miss.y_test)

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
#scr = mdl_miss.gpr.score(mdl_miss.X_test, mdl_miss.y_test)
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
        
scr = mdl_miss.o_ridge.score(mdl_miss.X_test, mdl_miss.y_test)

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
# Tm, zeta plots