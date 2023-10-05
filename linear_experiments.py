#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:32:15 2023

@author: jessicaclark
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score

import time

class bin_objective_linear:
    def __init__(self, tr_x, tr_y, va_x, va_y, te_x, te_y, f):
        self.tr_x = tr_x
        self.tr_y = np.ravel(tr_y)
        self.va_x = va_x
        self.va_y = np.ravel(va_y)
        self.te_x = te_x
        self.te_y = np.ravel(te_y)
        self.f = f

    def __call__(self, trial):
        
        my_C = trial.suggest_float('my_C', 1e-5, 1e5, log=True)
        
        classifier_obj = LogisticRegression(solver = 'liblinear', 
                                            penalty = 'l2',
                                            C = my_C)
        
        clf = classifier_obj.fit(self.tr_x, self.tr_y)
        preds = clf.predict_proba(self.va_x)[:,1]
        preds_test = clf.predict_proba(self.te_x)[:,1]
        
        bin_score = roc_auc_score(self.va_y, preds)
        test_score = roc_auc_score(self.te_y, preds_test)

        self.f.write(str(trial.number)+','+str(bin_score)+','+str(test_score)+','+str(time.time())+','+
                str(my_C)+'\n')
                
        return bin_score

class num_objective_linear:
    def __init__(self, tr_x, tr_y, va_x, va_y, te_x, te_y, f):
        self.tr_x = tr_x
        self.tr_y = np.ravel(tr_y)
        self.va_x = va_x
        self.va_y = np.ravel(va_y)
        self.te_x = te_x
        self.te_y = np.ravel(te_y)
        self.f = f

    def __call__(self, trial):
        

        my_alpha = trial.suggest_float('my_alpha', 1e-5, 1e5, log=True)
        
        regressor_obj = Ridge(alpha = my_alpha)
        clf = regressor_obj.fit(self.tr_x, self.tr_y)
        preds = clf.predict(self.va_x)
        preds_test = clf.predict(self.te_x)
        
            
        num_score = r2_score(self.va_y, preds)
        test_score = r2_score(self.te_y, preds_test)

        self.f.write(str(trial.number)+','+str(num_score)+','+str(test_score)+','+str(time.time())+
                     ','+str(my_alpha)+'\n')
                
        return num_score


def run_linear_experiments(study, numtrials, targ_name, targ_type, sampler_type, seed_num, data_size, tr_x, va_x, tr_y, va_y, tr_y_bin, va_y_bin, X_big_te, yvar_te, yvar_te_bin):

        f = open('linear_'+targ_name+'_'+targ_type+'_'+sampler_type+'_'+str(data_size)+'_seed_'+str(seed_num)+'.txt', 'w')
        f.write('trial,score,test_score,time,alpha\n')
        f.write('0,0,0,'+str(time.time())+',0\n')

        if targ_type == 'bin':
            
            study.enqueue_trial({
                "my_C": 1
            })
            
            study.optimize(bin_objective_linear(tr_x, 
                                                tr_y_bin,
                                                va_x,
                                                va_y_bin,
                                                X_big_te,
                                                yvar_te_bin,
                                                f), n_trials=numtrials, timeout=14400)
            
        elif targ_type == 'num':
            
            study.enqueue_trial({
                "my_alpha":1
            })
            
            study.optimize(num_objective_linear(tr_x, 
                                             tr_y, 
                                             va_x, 
                                             va_y,
                                             X_big_te,
                                             yvar_te,
                                                f), n_trials=numtrials, timeout=14400)
            
        f.close()     