#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:10:54 2023

Authors: Dr. Jessica M Clark (jmclark@umd.edu) and Praharsh Deep Singh
"""

import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import time

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as ty
from torch.utils.data import DataLoader, TensorDataset


class ResNetBlock(nn.Module):
    def __init__(self, d: int, d_hidden: int, hidden_dropout: float, residual_dropout: float):
        super().__init__()
        self.norm = nn.BatchNorm1d(d)
        self.linear0 = nn.Linear(d, d_hidden)
        self.hidden_dropout = hidden_dropout
        self.linear1 = nn.Linear(d_hidden, d)
        self.residual_dropout = residual_dropout
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        z = self.norm(z)
        z = self.linear0(z)
        z = F.relu(z)
        if self.hidden_dropout:
            z = F.dropout(z, self.hidden_dropout, self.training)
        z = self.linear1(z)
        if self.residual_dropout:
            z = F.dropout(z, self.residual_dropout, self.training)
        x = x + z
        return x


class ResNetModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int,
        layer_size: int,
        hidden_factor: float,
        hidden_dropout: float,
        residual_dropout: float,
        category_embedding_size: int,
        categories: ty.Optional[ty.List[int]]
    ):
        super().__init__()

        # Handling categorical features if present
        if categories is not None:
            self.category_embeddings = nn.ModuleList([
                nn.Embedding(num_categories, category_embedding_size)
                for num_categories in categories
            ])
        else:
            self.category_embeddings = None
        if category_embedding_size is not None:
            self.first_layer = nn.Linear(input_size + (len(categories) * category_embedding_size), layer_size)
        else:
            self.first_layer = nn.Linear(input_size, layer_size)
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(layer_size, int(layer_size * hidden_factor), hidden_dropout, residual_dropout)
            for _ in range(num_layers)
        ])
        self.last_normalization = nn.BatchNorm1d(layer_size)
        self.last_activation = nn.ReLU()
        self.head = nn.Linear(layer_size, output_size)

    def forward(self, x_num: torch.Tensor, x_cat: ty.Optional[torch.Tensor]) -> torch.Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            category_embeddings = [
                emb(x_cat[:, i]) for i, emb in enumerate(self.category_embeddings)
            ]
            x.append(torch.cat(category_embeddings, dim=-1))
        
        x = torch.cat(x, dim=-1)
        x = self.first_layer(x)
        
        for block in self.resnet_blocks:
            x = block(x)
            
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        return x.squeeze(-1)  # Squeeze to match the expected output shape



    
    # run 15 experiments with different random seeds and report performance on a test set
# use same train/validation/test split for each    
class bin_objective_nn:
    
    def __init__(self, tr_x, tr_y, va_x, va_y, te_x, te_y, f, exp_ind):
        self.tr_x = tr_x
        self.tr_y = np.ravel(tr_y)
        self.va_x = va_x
        self.va_y = np.ravel(va_y)
        self.te_x = te_x
        self.te_y = np.ravel(te_y)
        self.f = f
        self.exp_ind = exp_ind
        
    def __call__(self, trial):
        
        nest_tr_x, nest_va_x, nest_tr_y, nest_va_y = train_test_split(self.tr_x, self.tr_y, test_size = .2)
        

        input_size = nest_tr_x.shape[1]
        num_classes = 2  # Binary classification

        # Hyperparameters - new version is different?
        num_layers = trial.suggest_int("num_layers", 5, 10)
        layer_size = trial.suggest_int("layer_size", 8, 16)
        hidden_factor = trial.suggest_float("hidden_factor", 1, 4)
        hidden_dropout = trial.suggest_float("hidden_dropout", 0, 0.5)
        residual_dropout = trial.suggest_float("residual_dropout", 0, 0.5, step=0.1)  # Use 0 as an option
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        category_embedding_size = trial.suggest_int("category_embedding_size", 64, 512)
    
        # Building model
        model = ResNetModel(input_size, num_classes, num_layers, layer_size, hidden_factor,
                            hidden_dropout, residual_dropout, category_embedding_size = None, categories = None)
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        patience = 16
        current_patience = patience
        best_auc = 0.0
        epochs_without_improvement = 0
    
        tr_x_coo = nest_tr_x.tocoo()
        tr_x_row = torch.tensor(tr_x_coo.row, dtype=torch.long)
        tr_x_col = torch.tensor(tr_x_coo.col, dtype=torch.long)
        tr_x_data = torch.tensor(tr_x_coo.data, dtype=torch.float)
        tr_x_tensor = torch.sparse_coo_tensor(torch.stack([tr_x_row, tr_x_col]), tr_x_data, size=nest_tr_x.shape)
        tr_y_tensor = torch.tensor(nest_tr_y, dtype=torch.long)
        va_x_coo = nest_va_x.tocoo()
        va_x_row = torch.tensor(va_x_coo.row, dtype=torch.long)
        va_x_col = torch.tensor(va_x_coo.col, dtype=torch.long)
        va_x_data = torch.tensor(va_x_coo.data, dtype=torch.float)
        va_x_tensor = torch.sparse_coo_tensor(torch.stack([va_x_row, va_x_col]), va_x_data, size=nest_va_x.shape)
    
        # Convert data to PyTorch datasets
        train_dataset = TensorDataset(tr_x_tensor, tr_y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        
        outer_va_x_coo = self.va_x.tocoo()
        outer_va_x_row = torch.tensor(outer_va_x_coo.row, dtype = torch.long)
        outer_va_x_col = torch.tensor(outer_va_x_coo.col, dtype=torch.long)
        outer_va_x_data = torch.tensor(outer_va_x_coo.data, dtype=torch.float)
        outer_va_x_tensor = torch.sparse_coo_tensor(torch.stack([outer_va_x_row, outer_va_x_col]), outer_va_x_data, size=self.va_x.shape)
        
        te_x_coo = self.te_x.tocoo()
        te_x_row = torch.tensor(te_x_coo.row, dtype=torch.long)
        te_x_col = torch.tensor(te_x_coo.col, dtype=torch.long)
        te_x_data = torch.tensor(te_x_coo.data, dtype=torch.float)
        te_x_tensor = torch.sparse_coo_tensor(torch.stack([te_x_row, te_x_col]), te_x_data, size=self.te_x.shape)
    
        
        epoch_counter = 0
    
        while(current_patience >= 0) and (epoch_counter < 200):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x, None)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    batch_y_tensor = torch.tensor(batch_y, dtype=torch.long).clone().detach()
                loss = criterion(outputs, batch_y_tensor)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                va_outputs = model(va_x_tensor, None)
            va_preds = torch.softmax(va_outputs, dim=1)[:, 1].numpy()
            roc_auc = roc_auc_score(nest_va_y, va_preds)
    

            # model.eval()
            # with torch.no_grad():
            #     va_outputs = model(va_x_tensor, None)
            #     va_preds = torch.softmax(va_outputs, dim=1)[:, 1].numpy()
            
            
            if roc_auc > best_auc:
                best_auc = roc_auc
                epochs_without_improvement = 0
                current_patience = patience
                torch.save(model.state_dict(), 'best_bin_model_'+str(self.exp_ind)+'.pth')
            else:
                epochs_without_improvement += 1
                current_patience -= 1
                
            epoch_counter+=1

            # Loading the model
        best_model = ResNetModel(input_size, num_classes, num_layers, layer_size, hidden_factor,
                                hidden_dropout, residual_dropout, category_embedding_size=None, categories=None)
        best_model.load_state_dict(torch.load('best_bin_model_'+str(self.exp_ind)+'.pth'))
        best_model.eval()
        with torch.no_grad():

            outer_va_outputs = best_model(outer_va_x_tensor, None)
            outer_va_probs = torch.softmax(outer_va_outputs, dim = 1)[:, 1].numpy()    
            outer_va_roc_auc = roc_auc_score(self.va_y, outer_va_probs)                
            
            te_outputs = best_model(te_x_tensor, None)
            te_probs = torch.softmax(te_outputs, dim=1)[:, 1].numpy()
            test_roc_auc = roc_auc_score(self.te_y, te_probs)
            
        print(str(trial.number)+' '+str(epoch_counter)+' '+str(best_auc)+' '+str(outer_va_roc_auc)+' '+str(test_roc_auc))

        self.f.write(f"{trial.number},{outer_va_roc_auc},{test_roc_auc},{time.time()},"
                f"{num_layers},{layer_size},{hidden_factor},{hidden_dropout},"
                f"{residual_dropout},{learning_rate},{weight_decay},"
                f"{category_embedding_size}\n")
        
        return outer_va_roc_auc



class num_objective_nn:
    def __init__(self, tr_x, tr_y, va_x, va_y, te_x, te_y, f, exp_ind):
        self.tr_x = tr_x
        self.tr_y = np.ravel(tr_y)
        self.va_x = va_x
        self.va_y = np.ravel(va_y)
        self.te_x = te_x
        self.te_y = np.ravel(te_y)
        self.f = f
        self.exp_ind = exp_ind
        
    def __call__(self, trial):
        
        nest_tr_x, nest_va_x, nest_tr_y, nest_va_y = train_test_split(self.tr_x, self.tr_y, test_size = .2)
        

        input_size = nest_tr_x.shape[1]
        output_size = 1  # Binary classification

    
        # Hyperparameters
        num_layers = trial.suggest_int("num_layers", 5, 10)
        layer_size = trial.suggest_int("layer_size", 8, 16)
        hidden_factor = trial.suggest_float("hidden_factor", 1, 4)
        hidden_dropout = trial.suggest_float("hidden_dropout", 0, 0.5)
        residual_dropout = trial.suggest_float("residual_dropout", 0, 0.5, step=0.1)  # Use 0 as an option
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        category_embedding_size = trial.suggest_int("category_embedding_size", 64, 512)
        
        # Building model
        model = ResNetModel(input_size, output_size, num_layers, layer_size, hidden_factor,
                            hidden_dropout, residual_dropout, category_embedding_size=None, categories=None)
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        patience = 16
        current_patience = patience
        best_r2 = float('-inf')
        epochs_without_improvement = 0
    
        tr_x_coo = nest_tr_x.tocoo()
        tr_x_row = torch.tensor(tr_x_coo.row, dtype=torch.long)
        tr_x_col = torch.tensor(tr_x_coo.col, dtype=torch.long)
        tr_x_data = torch.tensor(tr_x_coo.data, dtype=torch.float)
        tr_x_tensor = torch.sparse_coo_tensor(torch.stack([tr_x_row, tr_x_col]), tr_x_data, size=nest_tr_x.shape)
        tr_y_tensor = torch.tensor(nest_tr_y, dtype=torch.long)
        va_x_coo = nest_va_x.tocoo()
        va_x_row = torch.tensor(va_x_coo.row, dtype=torch.long)
        va_x_col = torch.tensor(va_x_coo.col, dtype=torch.long)
        va_x_data = torch.tensor(va_x_coo.data, dtype=torch.float)
        va_x_tensor = torch.sparse_coo_tensor(torch.stack([va_x_row, va_x_col]), va_x_data, size=nest_va_x.shape)
    
        # Convert data to PyTorch datasets
        train_dataset = TensorDataset(tr_x_tensor, tr_y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        
        outer_va_x_coo = self.va_x.tocoo()
        outer_va_x_row = torch.tensor(outer_va_x_coo.row, dtype = torch.long)
        outer_va_x_col = torch.tensor(outer_va_x_coo.col, dtype=torch.long)
        outer_va_x_data = torch.tensor(outer_va_x_coo.data, dtype=torch.float)
        outer_va_x_tensor = torch.sparse_coo_tensor(torch.stack([outer_va_x_row, outer_va_x_col]), outer_va_x_data, size=self.va_x.shape)
        
        te_x_coo = self.te_x.tocoo()
        te_x_row = torch.tensor(te_x_coo.row, dtype=torch.long)
        te_x_col = torch.tensor(te_x_coo.col, dtype=torch.long)
        te_x_data = torch.tensor(te_x_coo.data, dtype=torch.float)
        te_x_tensor = torch.sparse_coo_tensor(torch.stack([te_x_row, te_x_col]), te_x_data, size=self.te_x.shape)
    

      

    
        # Model Training
        
        epoch_counter = 0
        
        while(current_patience >= 0) and (epoch_counter<200):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x, None)
                batch_y_tensor = batch_y.unsqueeze(1).to(outputs.dtype).squeeze()
                loss = criterion(outputs, batch_y_tensor)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                va_outputs = model(va_x_tensor, None).numpy()
            r2 = r2_score(nest_va_y, va_outputs.flatten())
            
            if r2 > best_r2:
                best_r2 = r2
                epochs_without_improvement = 0
                current_patience = patience
                # Saving the model
                torch.save(model.state_dict(), 'best_num_model_'+str(self.exp_ind)+'.pth')
            else:
                epochs_without_improvement += 1
                current_patience -= 1
                
            epoch_counter+=1
        

        
        # Loading the model
        best_model = ResNetModel(input_size, output_size, num_layers, layer_size, hidden_factor,
                            hidden_dropout, residual_dropout, category_embedding_size=None, categories=None)
        best_model.load_state_dict(torch.load('best_num_model_'+str(self.exp_ind)+'.pth'))
        best_model.eval()
        with torch.no_grad():

            outer_va_outputs = best_model(outer_va_x_tensor, None).numpy()
            valid_r2 = r2_score(self.va_y, outer_va_outputs.flatten())            
            
            te_outputs = best_model(te_x_tensor, None).numpy()
            te_r2 = r2_score(self.te_y, te_outputs.flatten())
            
        print(str(trial.number)+' '+str(epoch_counter)+' '+str(r2)+' '+str(valid_r2)+' '+str(te_r2))

        self.f.write(f"{trial.number},{valid_r2},{te_r2},{time.time()},"
                f"{num_layers},{layer_size},{hidden_factor},{hidden_dropout},"
                f"{residual_dropout},{learning_rate},{weight_decay},"
                f"{category_embedding_size}\n")
        
        return valid_r2


    

    
#need defaults
def run_nn_experiments(study, numtrials, targ_name, targ_type, sampler_type, seed_num, data_size, 
                       tr_x, va_x, tr_y, va_y, tr_y_bin, va_y_bin, X_big_te, yvar_te, yvar_te_bin, 
                       exp_ind):
    
    
    f = open('nn_'+targ_name+'_'+targ_type+'_'+sampler_type+'_'+str(data_size)+'_seed_'+str(seed_num)+'.txt', 'w')
    f.write('trial,score,test_score,time,num_layers, layer_size, hidden_factor, hidden_dropout, residual_dropout, learning_rate, weight_decay, category_embedding_size\n')
    f.write('0,0,0,'+str(time.time())+'0,0,0,0,0,0,0,0\n')
       
        
    #defaults
    study.enqueue_trial({
        
        "num_layers": 5,
        "layer_size": 8,
        "hidden_factor": 3,
        "hidden_dropout": 0.4,
        "residual_dropout": 0.4,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "category_embedding_size": 64
      
    })
       
       
    if targ_type == 'bin':
        study.optimize(bin_objective_nn(tr_x, 
                                     tr_y_bin, 
                                     va_x, 
                                     va_y_bin,
                                     X_big_te,                                                     
                                     yvar_te_bin,
                                     f,
                                     exp_ind), n_trials=numtrials, timeout=600000)
       
    elif targ_type == 'num':
        study.optimize(num_objective_nn(tr_x, 
                                     tr_y, 
                                     va_x, 
                                     va_y,
                                     X_big_te,
                                     yvar_te,
                                     f,
                                     exp_ind), n_trials=numtrials, timeout=600000)
                        
       
    f.close()