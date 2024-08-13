#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:49:56 2024

@author: sascha
"""

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
numpyro.set_host_device_count(4)
from jax import random
import jax

# Ensure JAX uses all available CPU cores
jax.config.update('jax_platform_name', 'cpu')
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
# numpyro.set_host_device_count(4)

'''
    Individual participants, all jokertypes
    With v ~ 1 + jokercondition
    i.e., not hierarchically modelled
'''

# ddmdata = ddmdata[ddmdata['jokercondition'] == 2]
# time.sleep(36_000)

R_thresh = 1.05

initvalues = {'a': 0.8, 
              't': 0.05,
              'theta': 0.2, 
              'z_Intercept': 0.5,
              'v_Intercept': 2.0,
              'v_qdiff': 2,
              'v_repdiff': 2,
              'z_repdiff|jokercondition': [0, 0, 0,]}

for day in range(2, 3):
    ddmdata = DDMutils.get_DDM_data2(day, RL=True)
    
    ddmdata.rename(columns={'subject': 'participant_id'}, inplace = True)
    ddmdata['min'] = ddmdata.apply(lambda row: min(abs(row['qdiff']), abs(row['repdiff'])), axis = 1)

    n_draws = 2_0
    n_tune = 2_0
    repeat = 1
    num_rep = 1
    
    while repeat and num_rep < 4:        
        print(f"\n\nInference for day {day}. Repetition number {num_rep}.")    
        model_reg_v_angle_hier = hssm.HSSM(
            data = ddmdata,
            model = "angle",
            hierarchical = True,
            categorical = 'jokercondition',
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 1 + qdiff + (1 + qdiff|participant_id) + repdiff + (1 + repdiff|participant_id) + ( 1 + min|participant_id)",
                    # "formula": "v ~ log_transform(qdiff)",
                    "prior": {
                        "Intercept": {"name": "Normal", 
                                      "mu": 1, 
                                      "sigma": 2, 
                                      "initval": initvalues['v_Intercept']},
                        "qdiff": {"name": "Normal", 
                                      "mu": 0, 
                                      "sigma": 2, 
                                      "initval": initvalues['v_qdiff']},
                        "repdiff": {"name": "Normal", 
                                      "mu": 0, 
                                      "sigma": 2, 
                                      "initval": initvalues['v_repdiff']},
                    },
                    "link": "identity",
                },
                {
                    "name": "z",
                    "formula": "z ~ 1 + (0 + repdiff|jokercondition)",
                    "prior": {
                        "Intercept": {
                            "name": "Uniform",
                            "lower": 0.3,
                            "upper": 0.7,
                            "initval": initvalues['z_Intercept']},
                        # "repdiff|jokercondition": {"name": "Normal", 
                        #                       "mu": {"name": "Normal", "mu": 0, "sigma": 0.5},
                        #                       "sigma": {"name": "HalfNormal", "sigma": .3}}
                    },
                    "link": "identity",
                },
            ],
        )

        infer_data_trace = model_reg_v_angle_hier.sample(
            sampler="nuts_numpyro", chains=4, cores=4, draws=n_draws, tune=n_tune,
            include_mean = False)

        # summary = az.summary(infer_data_trace, var_names = ['a',
        #                                                     't',
        #                                                     'theta',
        #                                                     'v_Intercept',
        #                                                     'v_min',
        #                                                     'v_qdiff',
        #                                                     'v_repdiff',
        #                                                     'z_Intercept',
        #                                                     'z_1|jokercondition',
        #                                                     'z_repdiff|jokercondition',
        #                                                     'z_repdiff|jokercondition_mu',
        #                                                     'z_repdiff|jokercondition_sigma'])

        summary = model_reg_v_angle_hier.summary()

        print(summary)

        if np.any(summary['r_hat']>R_thresh):
            n_draws += 2000
            n_tune += 2000
            num_rep += 1
            
            if np.any(summary['r_hat']<=R_thresh):
                summary = summary[summary['r_hat']<=R_thresh]
                
                for param in summary.index:
                    if 'z_1|jokercondition' not in param and  'z_repdiff|jokercondition' not in param:
                        print(f"Setting {param}")
                        initvalues[param] = summary.loc[param, 'mean']
                        
                    elif 'z_1|jokercondition[' in param:
                        index = int(param[-4])-1
                        print(f"Setting {param} (index {index})")
                        initvalues['z_1|jokercondition'][index] = summary.loc[param, 'mean']
                        
                    elif 'z_repdiff|jokercondition[' in param:
                        index = int(param[-4])-1
                        print(f"Setting {param} (index {index})")
                        initvalues['z_repdiff|jokercondition'][index] = summary.loc[param, 'mean']

        else:
            repeat = 0

            print(f"Saving Results for pid {pid}.")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            pickle.dump( summary, 
                        open(f"RLmodel/{timestamp}_Day{day}_pid{pid}.p", "wb" ) )
            
            print("== == == == ==")

#%%
if 0:            
    summary_all = az.summary(infer_data_trace) 
    
    
    for rowidx in range(len(summary_all)):
        if 'a' in summary_all.index[rowidx] and summary_all.index[rowidx][0:2] != 'a[' and summary_all.index[rowidx][0:2] != 'z_' and summary_all.index[rowidx][0:2] != 'v_':
            print(summary_all.index[rowidx])
    
    print("\n\nPrinting mu\n")    
    for rowidx in range(len(summary_all)):
        if 'mu' in summary_all.index[rowidx]:
            print(summary_all.index[rowidx])
            
    print("\n\nPrinting sigma\n")    
    for rowidx in range(len(summary_all)):
        if 'sigma' in summary_all.index[rowidx]:
            print(summary_all.index[rowidx])
            
    print("\n\nPrinting v_\n")    
    for rowidx in range(len(summary_all)):
        if 'v_' in summary_all.index[rowidx]:
            print(summary_all.index[rowidx])
            
    print("\n\nPrinting z_\n")    
    for rowidx in range(len(summary_all)):
        if 'z_' in summary_all.index[rowidx]:
            print(summary_all.index[rowidx])
        
    print("\n\nPrinting theta\n")    
    for rowidx in range(len(summary_all)):
        if 'theta' in summary_all.index[rowidx]:
            print(summary_all.index[rowidx])
    