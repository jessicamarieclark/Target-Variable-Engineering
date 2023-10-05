#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:01:05 2023

@author: jessicaclark
"""

import pandas as pd
import scipy.sparse as spse
from sklearn.preprocessing import StandardScaler
import numpy as np



def sparse_loader(infile_prefix, filename, datasize):
    
    if datasize <= 100:
        appendage = filename+str(datasize)
    else:
        appendage = filename
    
    features = pd.read_csv(infile_prefix+"feats_tabular_"+appendage+".csv", header = 0)
        
    features[["rownum"]] = features[["rownum"]] - 1
    features[["col"]] = features[["col"]] - 1
    features[["x"]] = 1
    
    data_arr = features["x"].to_numpy()
    row_arr = features["rownum"].to_numpy()
    col_arr = features["col"].to_numpy().astype(int)
    
    #figure out shape of data  
    row_max = max(row_arr)
    col_max = max(col_arr)
    data_shape = (row_max+1, col_max+1)
    
    sparse_features = spse.coo_matrix((data_arr, (row_arr, col_arr)), shape = data_shape)
    sparse_features = spse.csc_matrix(sparse_features)
    scaler = StandardScaler(with_mean = False)
    sparse_features = scaler.fit_transform(sparse_features)
    sparse_features = sparse_features.asfptype()
    
    return sparse_features

def load_sparse_data(folder_name, y_name, datasize):
    
    infile_prefix = 'tve_data/'+folder_name+'/'
    
    targets_tr = pd.read_csv(infile_prefix+'targs_tabular_tr.csv', sep = ',')
    yvar_tr = targets_tr[[y_name]].to_numpy()
    yvar_tr[np.isnan(yvar_tr)] = 0
    
    targets_te = pd.read_csv(infile_prefix+'targs_tabular_te.csv', sep = ',')
    yvar_te = targets_te[[y_name]].to_numpy()
    yvar_te[np.isnan(yvar_te)] = 0
      
    all_yvar = yvar_tr.ravel().tolist()+yvar_te.ravel().tolist()
    all_yvar = np.array(all_yvar)
   
    #standardize y variable
    yvar_tr = (yvar_tr - np.mean(all_yvar))/np.std(all_yvar)
    yvar_te = (yvar_te - np.mean(all_yvar))/np.std(all_yvar)
    
    #binarize y variable
    yvar_tr_bin = np.where(yvar_tr <= 0, 0, 1)
    yvar_te_bin = np.where(yvar_te <= 0, 0, 1)
        
    features_tr = sparse_loader(infile_prefix, 'tr', datasize)
    features_te = sparse_loader(infile_prefix, 'te', datasize)
      
    return features_tr, features_te, yvar_tr, yvar_te, yvar_tr_bin, yvar_te_bin

load_sparse_data('yelp', 'total_reviews', 2000)