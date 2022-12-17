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
idx = pd.IndexSlice
pd.options.display.max_rows = 30

# temporary spyder debugger error hack
import collections
collections.Callable = collections.abc.Callable

#%% concat with other data
loss_data = pd.read_csv('./results/loss_estimate_data.csv', index_col=None)
full_isolation_data = pd.read_csv('full_isolation_data.csv', index_col=None)

df = pd.concat([full_isolation_data, loss_data], axis=1)
# df['max_drift'] = df[["driftMax1", "driftMax2", "driftMax3"]].max(axis=1)
# df['max_accel'] = df[["accMax0", "accMax1", "accMax2", "accMax3"]].max(axis=1)
# df['max_vel'] = df[["velMax0", "velMax1", "velMax2", "velMax3"]].max(axis=1)
#%% new class
class Prediction:
    
    def __init__(self, data, outcome):
        self._raw_data = data
        self.k = len(data)
        self.X = data[['gapRatio', 'RI', 'zetaM', 'Tm']]
        self.y = data[[outcome]]
        
    # TODO: scale data
    # consider method that cleans data based on prediction outcome
        
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
            {'svr__C':[0.1, 1.0, 10.0, 100.0],
             'svr__epsilon':[0.01, 0.1, 1.0],
             'svr__gamma':np.logspace(-2, 2, 5)}
            ]
        
        svr_cv = GridSearchCV(sv_pipe, param_grid=parameters)
        svr_cv.fit(self.X_train, self.y_train)
        
        # set pipeline to use CV params
        sv_pipe.set_params(**svr_cv.best_params_)
        sv_pipe.fit(self.X_train, self.y_train)
        
        self.svr = sv_pipe
        
        # ultimately, want:
            # fit method to apply to new data
            # details of params used
            # training score
            # testing score
     
    def fit_svc(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        
        # pipeline to scale -> SVC
        sv_pipe = Pipeline([('scaler', StandardScaler()),
                            ('svc', SVC(kernel='rbf', gamma='auto'))])
        
        # cross-validate several parameters
        parameters = [
            {'svc__C':[0.1, 1.0, 10.0, 100.0]}
            ]
        
        svc_cv = GridSearchCV(sv_pipe, param_grid=parameters)
        svc_cv.fit(self.X_train, self.y_train)
        
        # set pipeline to use CV params
        sv_pipe.set_params(**svc_cv.best_params_)
        sv_pipe.fit(self.X_train, self.y_train)
        
        print("The best parameters are %s with a score of %0.2f"
              % (svc_cv.best_params_, svc_cv.best_score_))
        
    # TODO: kernel ridge
        
# TODO: fit impact

#%% fit impact
mdl = Prediction(df, 'impacted')
mdl.test_train_split(0.2)
mdl.fit_svc()
#%% perform fits

mdl = Prediction(df, 'cost_50%')
mdl.test_train_split(0.2)

# SVR is currently a poor fit, investigate
mdl.fit_svr()
preds = mdl.svr.predict(mdl.X_test)

# possible checks:
    # transform, should apply to test as well?
    # predict impact along with set?
    
# TODO: generate prediction grid and plot