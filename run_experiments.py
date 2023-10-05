#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:56:12 2023

@author: jessicaclark
"""


import sys
import optuna
from sklearn.model_selection import train_test_split

import sparse_data_loading
import running_params
import xgboost_experiments
import linear_experiments
import resnet_experiments


numtrials = 400


def set_up_experiments(params, exp_ind):
    
    folder_name = params[0]
    targ_name = params[1]
    data_size = params[2]
    model_type = params[3]
    targ_type = params[4]
    sampler_type = params[5]
    seed_num = params[6]
    
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
      
    
    X_big_tr, X_big_te, yvar_tr, yvar_te, yvar_tr_bin, yvar_te_bin = sparse_data_loading.load_sparse_data(folder_name, targ_name, data_size)
    

    #separate validation data from training data
    tr_x, va_x, tr_y, va_y, tr_y_bin, va_y_bin = train_test_split(X_big_tr, yvar_tr, yvar_tr_bin, 
                                                                  test_size = .33, 
                                                                  random_state = 1)
    
    if sampler_type == 'random':
        experiment_sampler = optuna.samplers.RandomSampler(seed=seed_num)
    elif sampler_type == 'tpe':
        experiment_sampler = optuna.samplers.TPESampler(seed=seed_num)
                    
    study = optuna.create_study(direction="maximize",sampler=experiment_sampler)    
    
    
    if model_type == 'xgb':
        xgboost_experiments.run_xgb_experiments(study, numtrials, targ_name, targ_type, sampler_type, seed_num, data_size,
                            tr_x, va_x, tr_y, va_y, tr_y_bin, va_y_bin, X_big_te, yvar_te, yvar_te_bin,
                            exp_ind)
    elif model_type == 'linear':
        linear_experiments.run_linear_experiments(study, numtrials, targ_name, targ_type, sampler_type, seed_num, data_size,
                            tr_x, va_x, tr_y, va_y, tr_y_bin, va_y_bin, X_big_te, yvar_te, yvar_te_bin,
                            exp_ind)                                             
    elif model_type == 'nn':
        resnet_experiments.run_nn_experiments(study, numtrials, targ_name, targ_type, sampler_type, seed_num, data_size,
                            tr_x, va_x, tr_y, va_y, tr_y_bin, va_y_bin, X_big_te, yvar_te, yvar_te_bin,
                            exp_ind)

 

def main():
    
    param_list = running_params.create_param_array()
    operate_ind = int(sys.argv[1])-1
    set_up_experiments(param_list[operate_ind], operate_ind)
    

if __name__ == "__main__":
    main()
