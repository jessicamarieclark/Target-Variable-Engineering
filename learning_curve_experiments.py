#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:14:14 2023

Authors: Dr. Jessica M Clark and Praharsh Deep Singh
"""

import pandas as pd
import scipy.sparse as spse
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score



import xgboost as xgb

import sparse_data_loading

result_folder_name = ''
hp_choice_type = 'best'
runs = ''

def get_hyperparams(result_folder_name, params, targ_type, hp_type):
    
    targ_name = params[0]
    data_size = params[1]
    model_type = params[2]
    targ_type = params[3]
    sampler_type = params[4]
    seed_num = params[5]
    

    
    if hp_type == 'best':     
        
        filename = result_folder_name+targ_name+'_'+targ_type+'_'+sampler_type+'_'+str(data_size)+'_seed_'+str(seed_num)+'.txt'
        
        file = pd.read_csv(filename)
        
        max_row_num = file['score'].argmax()    
        best_row = file.iloc[max_row_num]

    
        param = {
            #fixed
            "verbosity": 0,
            "tree_method": "auto",
    
            #fixed, per paper
            "booster": "gbtree",
            #"early-stopping-rounds": 50, need to implement by hand?
            "num_round": 2000,
    
    
            "max_depth": int(best_row['max_depth']),
            "min_child_weight": best_row['min_child_weight'],
            "subsample": best_row['subsample'],
            "eta": best_row['eta'],
            "colsample_bylevel": best_row['colsample_bylevel'],
            "colsample_bytree": best_row['colsample_bytree'],
            "gamma": best_row['gamma'],
            "lambda": best_row['lambda'],
            "alpha": best_row['alpha']
        } 
        
    elif hp_type == 'default':
        param = {
            #fixed
            "verbosity": 0,
            "tree_method": "auto",
    
            #fixed, per paper
            "booster": "gbtree",
            #"early-stopping-rounds": 50, need to implement by hand?
            "num_round": 2000,
    
    
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 1,
            "eta": .3,
            "colsample_bylevel": 1,
            "colsample_bytree": 1,
            "gamma": 1e-20,
            "lambda": 1,
            "alpha": 1e-20
        } 
    
    if targ_type == 'num':
        param['objective'] = "reg:squarederror"
    else:
        param['objective'] = "binary:logistic"
    
    
    return param

def num_model(X_tr, X_te, y_tr, y_te, ybin_tr, ybin_te, param):
    
    dtrain = xgb.DMatrix(X_tr, label = y_tr)
    dtest = xgb.DMatrix(X_te, label = y_te)
    
    bst = xgb.train(param, dtrain)
    
   
    preds_test = bst.predict(dtest)
    preds_test = np.ravel(preds_test)

    return preds_test

    
def bin_model(X_tr, X_te, y_tr, y_te, ybin_tr, ybin_te, param):
    
    dtrain = xgb.DMatrix(X_tr, label = y_tr)
    dtest = xgb.DMatrix(X_te, label = y_te)
    
    bst = xgb.train(param, dtrain)
    
    preds_test = bst.predict(dtest) 

    return preds_test



def get_performance(tr_x_temp, X_big_te, tr_y_temp, yvar_te, tr_y_bin_temp, yvar_te_bin, num_best_params, bin_best_params):
    num_best = num_model(tr_x_temp, X_big_te, tr_y_temp, yvar_te, tr_y_bin_temp, yvar_te_bin, num_best_params)
    bin_best = bin_model(tr_x_temp, X_big_te, tr_y_temp, yvar_te, tr_y_bin_temp, yvar_te_bin, bin_best_params)

    num_accuracy = r2_score(yvar_te, num_best)

    bin_accuracy = roc_auc_score(yvar_te_bin, bin_best)
    
    perf = [num_accuracy, bin_accuracy]
    
    return perf


def make_learning_curve(result_folder_name, feat_set_name, var_name, para, hp_choice_type, num_runs):
    
    print(var_name)
    

    num_best_params = get_hyperparams(result_folder_name, para, 'num', hp_choice_type)
    bin_best_params = get_hyperparams(result_folder_name, para, 'bin', hp_choice_type)
    
    
    tr_x, X_big_te, tr_y, yvar_te, tr_y_bin, yvar_te_bin = sparse_data_loading.load_sparse_data(result_folder_name, var_name, para[1])
    

    train_sizes = [100, 200, 400, 750, 1000, 2500, 5000, 6000, 8000, 10000, 15000, 20000]
    
    
    all_x = spse.vstack((tr_x, X_big_te))
    all_y = np.concatenate((tr_y, yvar_te),axis = 0)
    
    
    all_bin = np.concatenate((tr_y_bin, yvar_te_bin),axis = 0)    
    tr_x, X_big_te, tr_y, yvar_te, tr_y_bin, yvar_te_bin = train_test_split(all_x, all_y, all_bin, test_size = 5000, random_state = 1) 

    f = open(result_folder_name+'learningcurve_'+feat_set_name+'_'+var_name+'.csv', 'w')
    f.write('folder,var,train_size,index,num,bin\n')
        
    for train_size in train_sizes:
        
        print(train_size)

        
        for index in range(num_runs):
            # Randomly select train_size
            tr_inds = np.random.choice(tr_x.shape[0], train_size, replace=False)
            tr_x_temp = tr_x[tr_inds]
            tr_y_temp = tr_y[tr_inds]
            tr_y_bin_temp = tr_y_bin[tr_inds]

            perf = get_performance(tr_x_temp, X_big_te, tr_y_temp, yvar_te, tr_y_bin_temp, yvar_te_bin, num_best_params, bin_best_params)
            
            f.write(feat_set_name+','+var_name+','+str(train_size)+','+str(index)+','+str(perf[0])+','+str(perf[1])+'\n')


def main():
    
    
    model_type = 'xgb'
    targ_type = 'num'
    sampler_type = 'random'
    seed_num = 0
    
    feat_set_names = ['airbnb', 'kickstarter', 'yelp']
    
    kickstarter_targets = ['sentence_counter', 'avgsentencelength', 'goal', 'backers_count', 'pledged', 'num_rewards', 'min_reward',
                    'max_reward',  'sd_reward', 'processing_duration']
    airbnb_targets = ['accommodates', 'availability_30', 'availability_60', 'availability_90', 
                      'availability_365', 'host_listings_count', 'number_of_reviews', 
                      'price', 'review_scores_rating', 'reviews_per_month']
    yelp_targets = ['total_reviews', 'avg_stars', 'avg_useful', 'avg_funny', 'avg_cool', 
                                 'avg_user_review_count', 'pct_elite', 'pct_male', 'checkin_count', 'num_tips']
    
    
    for feat_set_name in feat_set_names:
        if feat_set_name == 'kickstarter':
            var_names = kickstarter_targets
            data_size = 2146
        if feat_set_name == 'airbnb':
            var_names = airbnb_targets
            data_size = 2046
        if feat_set_name == 'yelp':
            var_names = yelp_targets
            data_size = 2187
        for var_name in var_names:
            params = [var_name, data_size, model_type, targ_type, sampler_type, seed_num]

            make_learning_curve(feat_set_name, var_name, params, hp_choice_type, runs)
    
    
main()

