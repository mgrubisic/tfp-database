############################################################################
#               ML prediction models for isolator loss data

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: December 2022

# Description:  ML models

# Open issues:  (1) 

############################################################################

import pandas as pd
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
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    # TODO: svr, pipeline with scaler -> CV -> prediction
    
    # TODO: kernel ridge
        

#%% test

mdl = Prediction(df, 'cost_50%')
mdl.test_train_split(0.2)