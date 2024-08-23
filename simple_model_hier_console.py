#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:07:50 2024

@author: sascha
"""

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
numpyro.set_host_device_count(4)
from jax import random
import jax

# Ensure JAX uses all available CPU cores
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_enable_x64', True)

import time
from datetime import datetime
import pickle
import numpy as np
import matplotlib as plt
import hssm
import tqdm as notebook_tqdm
import os
import pandas as pd
import DDMutils
import bambi as bmb
import arviz as az

hssm.set_floatX("float64")

'''
    Individual participants, all jokertypes
    With v ~ 1 + jokercondition
    i.e., not hierarchically modelled
    
    1 = random
    2 = congruent
    3 = incongruent
'''

# ddmdata = ddmdata[ddmdata['jokercondition'] == 2]
# time.sleep(36_000)

num_agents = 60

initvalues = {'a_Intercept': 0.8, 
              'a_1|participant_id_mu': 0,
              'a_1|participant_id': [0]*num_agents,
              
              't_Intercept': 0.2, 
              't_1|participant_id_mu': 0,
              't_1|participant_id': [0]*num_agents,
              
              'theta_Intercept': 0.9, 
              'theta_1|participant_id_mu': 0,
              'theta_1|participant_id': [0]*num_agents,
              
              'z_Intercept': 0.5,
              'z_jokercondition': [0., 0.],
              'z_jokercondition|participant_id_mu': [0., 0.],
              'z_1|participant_id_mu': 0,
              'z_1|participant_id': [0]*num_agents,
              
              'v_Intercept': 2.0,
              'v_jokercondition': [0., 0],
              'v_1|participant_id_mu': 0,
              'v_1|participant_id': [0]*num_agents} # a, p, t, v, z

R_thresh = 1.05

for day in range(2, 3):
    ddmdata = DDMutils.get_DDM_data2(day)
    # ddmdata = ddmdata.drop(columns=['qdiff', 'repdiff'], axis = 1)
    
    ddmdata = ddmdata[ddmdata['participant_id'] <= num_agents]
    
    # ddmdata['jokercondition'] = ddmdata['jokercondition'].apply(lambda x: 'A' if x == 1 else 'B' if x == 2 else 'C' if x == 3 else None)
    
    ddmdata['participant_id'] = ddmdata['participant_id'].astype("category")
    ddmdata['jokercondition'] = ddmdata['jokercondition'].astype("category")
    
    ddmdata['repdiff'] = ddmdata['repdiff'].map(lambda x: "A" if x > 0 else 
                                                          "B" if x < 0 else "C")
    
    # ddmdata['dummyrand'] = ddmdata['jokercondition'].map(lambda x: 1 if x==1 else 0)
    ddmdata['dummycong'] = ddmdata['jokercondition'].map(lambda x: 1 if x==2 else 0)
    ddmdata['dummyinc'] = ddmdata['jokercondition'].map(lambda x: 1 if x==3 else 0)
    
    # ddmdata['testcolumn'] = ddmdata['participant_id'].map(lambda x: x*2)
    # ddmdata['testcolumn'] = np.random.choice([1, 2, 3], p = [0.33, 0.33, 0.34])
    
    n_draws = 2_000
    n_tune = 2_000
    repeat = 1
    num_rep = 1
    
    while repeat:
        hssm_model = hssm.HSSM(
            data = ddmdata,
            model = "angle",
            hierarchical = True,
            categorical = ['participant_id'],
            p_outlier = {"name": "Uniform", "lower": 0.0001, "upper": 0.5},
            lapse = bmb.Prior("Uniform", lower=0.0, upper=20.0),
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 1 + \
                                    (dummycong|participant_id) + \
                                    (dummyinc|participant_id)",
                    "prior": {
                        "Intercept": {"name": "Normal", 
                                      "mu": 1, 
                                      "sigma": 2, 
                                      "initval": initvalues['v_Intercept']},
                    },
                    "link": "identity",
                },
                {
                    "name": "z",
                    "formula": "z ~ 1 + \
                                (dummycong|participant_id) + \
                                (dummyinc|participant_id)",
                    "prior": {
                        "Intercept": {
                            "name": "Uniform",
                            "lower": 0.1,
                            "upper": 0.9,
                            "initval": initvalues['z_Intercept']},
                        
                    },
                    "link": "identity",
                },
                {
                    "name": "a",
                    "formula": "a ~  1 + (1|participant_id)",
                    "prior": {
                        "Intercept": {
                            "name": "Normal",
                            "mu": 0.8,
                            "sigma": 0.1,
                            "initval": initvalues['a_Intercept']},
                        
                        "1|participant_id": {"name": "Normal", 
                                              "mu": {"name": "Normal", 
                                                      "mu": 0., 
                                                      "sigma": 1,
                                                      "initval": initvalues['a_1|participant_id_mu']},
                                              "sigma": {"name": "HalfNormal", 
                                                        "sigma": .3},
                                              "initval": initvalues['a_1|participant_id']},
                    },
                    "link": "identity",
                },
                {
                    "name": "t",
                    "formula": "t ~  1 + (1|participant_id)",
                    "prior": {
                        "Intercept": {
                            "name": "Normal",
                            "mu": 0.25,
                            "sigma": 0.05,
                            "initval": initvalues['t_Intercept']},
                        
                        "1|participant_id": {"name": "Normal", 
                                              "mu": {"name": "Normal", 
                                                      "mu": 0, 
                                                      "sigma": 0.5},
                                              "sigma": {"name": "HalfNormal", 
                                                        "sigma": .3}},
                    },
                    "link": "identity",
                },
                {
                    "name": "theta",
                    "formula": "theta ~  1 + (1|participant_id)",
                    "prior": {
                        "Intercept": {
                            "name": "Normal",
                            "mu": 1,
                            "sigma": 0.3,
                            "initval": initvalues['theta_Intercept']},
                        
                        "1|participant_id": {"name": "Normal", 
                                              "mu": {"name": "Normal", 
                                                      "mu": 0, 
                                                      "sigma": 0.5},
                                              "sigma": {"name": "HalfNormal", 
                                                        "sigma": .3}},
                    },
                    "link": "identity",
                },
            ],
            )

        infer_data_trace = hssm_model.sample(
            sampler="nuts_numpyro", chains=4, cores=4, draws=n_draws, tune=n_tune,
            include_mean = False
            )
        
        summary = hssm_model.summary()
        
        print("Printing summary. \n\n")
        print(summary)
        
        if np.any(summary['r_hat'] > R_thresh) or np.any(np.isnan(summary['r_hat'])):
            
            n_draws += 2_000
            n_tune += 2_000
            num_rep += 1
            
            '''
                Set initial values of those parameters with rhat < 1.05
            '''
            if np.any(summary['r_hat'] <= R_thresh):
                print("\n\n Setting initial values\n\n")
                converged_summary = summary[summary['r_hat'] <= R_thresh]
                
                for param in converged_summary.index:
                    if '[' not in param:
                        print(f"Setting initvalues[{param}] to {summary.loc[param, 'mean']}.\n\n")
                        initvalues[param] = converged_summary.loc[param, 'mean']
                    
                    elif '[' in param:
                        
                        if 'jokercondition|participant_id' in param:
                            pass
                        
                        else:
                            variable = param.split('[')[0]
                            index = int(param.split('[')[1][0:-1])
                            
                            if '|participant_id' in param:
                                index -= 1
                                
                            else:
                                index -= 2
                                
                            print(f"Setting initvalues[{variable}][{index}] (param {param}) to {converged_summary.loc[param, 'mean']}.\n\n")
                            initvalues[variable][index] = converged_summary.loc[param, 'mean']
                        
                    else: 
                        print(f"Not setting {param}] :(")

                del summary
                del converged_summary
                del infer_data_trace
                del hssm_model

        elif np.all(summary['r_hat'] <= R_thresh):
            repeat = 0

            print("Saving Results.")
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            pickle.dump( summary, 
                        open(f"Hierarchical_model_{timestamp}_Day{day}.p", "wb" ) )
        
        print("== == == == ==")

summary = az.summary(infer_data_trace)        

#%%
for rowidx in range(len(summary)):
    if 'a' in summary.index[rowidx] and summary.index[rowidx][0:2] != 'a[' and summary.index[rowidx][0:2] != 'z_' and summary.index[rowidx][0:2] != 'v_':
        print(summary.index[rowidx])

print("\n\nPrinting mu\n")    
for rowidx in range(len(summary)):
    if 'mu' in summary.index[rowidx]:
        print(summary.index[rowidx])
        
print("\n\nPrinting sigma\n")    
for rowidx in range(len(summary)):
    if 'sigma' in summary.index[rowidx]:
        print(summary.index[rowidx])
        
print("\n\nPrinting v_\n")    
for rowidx in range(len(summary)):
    if 'v_' in summary.index[rowidx]:
        print(summary.index[rowidx])
        
print("\n\nPrinting z_\n")    
for rowidx in range(len(summary)):
    if 'v_' in summary.index[rowidx]:
        print(summary.index[rowidx])
    
