#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:28:54 2023

@author: jessicaclark
"""

import xgboost as xgb

from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score

import time

class bin_objective_xgb:
    def __init__(self, tr_x, tr_y, va_x, va_y, te_x, te_y, f):
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.va_x = va_x
        self.va_y = va_y
        self.te_x = te_x
        self.te_y = te_y
        self.f = f

    def __call__(self, trial):
        dtrain = xgb.DMatrix(self.tr_x, label = self.tr_y)
        dvalid = xgb.DMatrix(self.va_x, label = self.va_y)
        dtest = xgb.DMatrix(self.te_x, label = self.te_y)

        param = {
            #fixed
            "verbosity": 0,
            "objective": "binary:logistic", #change for numerical
            "tree_method": "auto",

            #fixed, per paper
            "booster": "gbtree",
            #"early-stopping-rounds": 50, need to implement by hand?
            "num_round": 1000,


            "max_depth": trial.suggest_int("max_depth", 1,11),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "eta": trial.suggest_float("eta", 1e-5, 1.0, log=True),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8,1e2,log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1e2, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1e2, log=True)
        }

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        bin_score = roc_auc_score(self.va_y, preds)
        
        preds_test = bst.predict(dtest)
        test_score = roc_auc_score(self.te_y, preds_test)
        self.f.write(str(trial.number)+','+str(bin_score)+','+str(test_score)+','+str(time.time())+','+str(param['max_depth'])+','+str(param['min_child_weight'])+','+
              str(param['subsample'])+','+str(param['eta'])+','+str(param['colsample_bylevel'])+','+
              str(param['colsample_bytree'])+','+str(param['gamma'])+','+str(param['lambda'])+','+str(param['alpha'])+'\n')
        return bin_score


class num_objective_xgb:
    def __init__(self, tr_x, tr_y, va_x, va_y, te_x, te_y, f):
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.va_x = va_x
        self.va_y = va_y
        self.te_x = te_x
        self.te_y = te_y
        self.f = f

    def __call__(self, trial):
        dtrain = xgb.DMatrix(self.tr_x, label = self.tr_y)
        dvalid = xgb.DMatrix(self.va_x, label = self.va_y)
        dtest = xgb.DMatrix(self.te_x, label = self.te_y)
    
        param = {
            #fixed
            "verbosity": 0,
            "objective": "reg:squarederror", #change for numerical
            "tree_method": "auto",

            #fixed, per paper
            "booster": "gbtree",
            #"early-stopping-rounds": 50, need to implement by hand?
            "num_round": 2000,


            "max_depth": trial.suggest_int("max_depth", 1,11),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "eta": trial.suggest_float("eta", 1e-5, 1.0, log=True),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8,1e2,log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1e2, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1e2, log=True)
        } 
    
        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid) #is this ok?
        num_score = r2_score(self.va_y, preds)
        
        preds_test = bst.predict(dtest)
        score_test = r2_score(self.te_y, preds_test)
        
        self.f.write(str(trial.number)+','+str(num_score)+','+str(score_test)+','+str(time.time())+','+str(param['max_depth'])+','+str(param['min_child_weight'])+','+
              str(param['subsample'])+','+str(param['eta'])+','+str(param['colsample_bylevel'])+','+
              str(param['colsample_bytree'])+','+str(param['gamma'])+','+str(param['lambda'])+','+str(param['alpha'])+'\n')
        return num_score
    
    
def run_xgb_experiments(study, numtrials, targ_name, targ_type, sampler_type, seed_num, data_size, tr_x, va_x, tr_y, va_y, tr_y_bin, va_y_bin, X_big_te, yvar_te, yvar_te_bin):
    
    f = open('xgb_'+targ_name+'_'+targ_type+'_'+sampler_type+'_'+str(data_size)+'_seed_'+str(seed_num)+'.txt', 'w')
    f.write('trial,score,test_score,time,max_depth,min_child_weight,subsample,eta,colsample_bylevel,colsample_bytree,gamma,lambda,alpha\n')
    f.write('0,0,0,'+str(time.time())+'0,0,0,0,0,0,0,0,0,0\n')
       
    study.enqueue_trial({
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1,
        "eta": .3,
        "colsample_bylevel": 1,
        "colsample_bytree": 1,
        "gamma": 1e-20,
        "lambda": 1,
        "alpha": 1e-20
    })
       
       
    if targ_type == 'bin':
        study.optimize(bin_objective_xgb(tr_x, 
                                     tr_y_bin, 
                                     va_x, 
                                     va_y_bin,
                                     X_big_te,                                                     
                                     yvar_te_bin,
                                         f), n_trials=numtrials, timeout=14400)
       
    elif targ_type == 'num':
        study.optimize(num_objective_xgb(tr_x, 
                                     tr_y, 
                                     va_x, 
                                     va_y,
                                     X_big_te,
                                     yvar_te,
                                         f), n_trials=numtrials, timeout=14400)
                        
       
    f.close()

