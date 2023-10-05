#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:19:06 2023

@author: jessicaclark
"""

def create_param_array():
    folder_names = ['airbnb', 'kickstarter', 'yelp']
    
    
    
    airbnb_var_names = ['accommodates', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'host_listings_count',
                 'number_of_reviews', 'price', 'review_scores_rating', 'reviews_per_month']
    kickstarter_var_names = ['sentence_counter', 'avgsentencelength', 'goal', 'backers_count', 'pledged', 'num_rewards', 'min_reward',
                'max_reward',  'sd_reward', 'processing_duration']
    yelp_var_names = ['total_reviews', 'avg_stars', 'avg_useful', 'avg_funny', 'avg_cool', 
                 'avg_user_review_count', 'pct_elite', 'pct_male', 'checkin_count', 'num_tips']
    
    
    data_sizes = [5, 100, 2000]
    targ_types = ['bin', 'num']
    model_types = ['linear', 'xgb', 'nn']
    sample_types = ['random', 'tpe']
    
    seeds = range(15)
    
    param_list = []
    for folder_name in folder_names:
        if folder_name == 'airbnb':
            var_names = airbnb_var_names
        elif folder_name == 'kickstarter':
            var_names = kickstarter_var_names
        elif folder_name == 'yelp':
            var_names = yelp_var_names
        for var_name in var_names:
            for data_size in data_sizes:
                for model_type in model_types:
                    for targ_type in targ_types:
                        for sampler_type in sample_types:
                            for seed in seeds:
                                param_list.append((folder_name,
                                                   var_name, 
                                                   data_size, 
                                                   model_type, 
                                                   targ_type, 
                                                   sampler_type, 
                                                   seed))
                                
    return param_list


