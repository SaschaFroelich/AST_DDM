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
    
    jokerconditions
    1 random
    2 congruent
    3 incongruent
'''

# ddmdata = ddmdata[ddmdata['jokercondition'] == 2]
# time.sleep(36_000)

R_thresh = 1.05
with_lapses = True

for day in range(2, 3):
    
    ddmdata = DDMutils.get_DDM_data2(day)
    
    initvalues = {'a': 0.8, 
                  't': 0.05,
                  'theta': 0.2, 
                  'z_Intercept': 0.5,
                  'v_Intercept': 2.0,
                  'v_jokercondition': [0, 0],
                  'z_jokercondition': [0, 0]}
    
    ddmdata.rename(columns={'subject': 'participant_id'}, inplace = True)
    
    for pid in range(1, 61):
        n_draws = 2_000
        n_tune = 2_000
        repeat = 1
        num_rep = 1
        
        while repeat and num_rep < 5:        
            print(f"Inference for pid {pid} on day {day}. Repetition number {num_rep}.")    
            ddmdata_onesub = ddmdata[ddmdata['participant_id'] == pid]

            if with_lapses:
                hssm_model = hssm.HSSM(
                    data = ddmdata_onesub,
                    model = "angle",
                    hierarchical = False,
                    categorical = 'jokercondition',
                    p_outlier = {"name": "Uniform", "lower": 0.0001, "upper": 0.5},
                    lapse = bmb.Prior("Uniform", lower=0.0, upper=20.0),
                    include = [
                        {
                            "name": "v",
                            "formula": "v ~ 1 + jokercondition",
                            "prior": {
                                "Intercept": {"name": "Normal", 
                                              "mu": 1, 
                                              "sigma": 2, 
                                              "initval": initvalues['v_Intercept']},
                                
                                "jokercondition": {"name": 
                                                   "Normal", 
                                                   "mu": 0, 
                                                   "sigma": 2, 
                                                   "initval": initvalues['v_jokercondition']},
                            },
                            "link": "identity",
                        },
                        {
                            "name": "z",
                            "formula": "z ~ 1 + jokercondition",
                            "prior": {
                                "Intercept": {
                                    "name": "Uniform",
                                    "lower": 0.3,
                                    "upper": 0.7,
                                    "initval": initvalues['z_Intercept']},
                                
                                "jokercondition": {"name": "Normal", 
                                                   "mu": 0, 
                                                   "sigma": 2, 
                                                   "initval": initvalues['z_jokercondition']},
                            },
                            "link": "identity",
                        }
                    ],
                )
                
            else:
                hssm_model = hssm.HSSM(
                    data = ddmdata_onesub,
                    model = "angle",
                    hierarchical = False,
                    categorical = 'jokercondition',
                    include=[
                        # {
                        #     "name": "a",
                        #     "formula": "a ~ 1 + (1|participant_id)",
                        #     "prior": {
                        #         "Intercept": {"name": "Normal", 
                        #                       "mu": 1, 
                        #                       "sigma": 2, 
                        #                       "initval": initvalues['a']},
                        #         # "jokercondition": {"name": "Normal", 
                        #         #                    "mu": 0, 
                        #         #                    "sigma": 2, 
                        #         #                    "initval": initvalues['v_jokercondition']},
                        #     },
                        #     "link": "identity",
                        # },
                        {
                            "name": "v",
                            "formula": "v ~ 1 + jokercondition",
                            "prior": {
                                "Intercept": {"name": "Normal", 
                                              "mu": 1, 
                                              "sigma": 2, 
                                              "initval": initvalues['v_Intercept']},
                                "jokercondition": {"name": "Normal", 
                                                   "mu": 0, 
                                                   "sigma": 2, 
                                                   "initval": initvalues['v_jokercondition']},
                            },
                            "link": "identity",
                        },
                        {
                            "name": "z",
                            "formula": "z ~ 1 + jokercondition",
                            "prior": {
                                "Intercept": {
                                    "name": "Uniform",
                                    "lower": 0.3,
                                    "upper": 0.7,
                                    "initval": initvalues['z_Intercept']},
                                "jokercondition": {"name": "Normal", 
                                                   "mu": 0, 
                                                   "sigma": 2, 
                                                   "initval": initvalues['z_jokercondition']},
                            },
                            "link": "identity",
                        }
                    ],
                )

            infer_data_trace = hssm_model.sample(
                sampler="nuts_numpyro", chains=4, cores=4, draws=n_draws, tune=n_tune,
                include_mean = False
                )
            
            summary = hssm_model.summary()
            print(summary)
            
            if np.any(summary['r_hat'] > R_thresh):
                n_draws += 2500
                n_tune += 2500
                num_rep += 1
                
                if np.any(summary['r_hat'] <= R_thresh):
                    summary = summary[summary['r_hat'] <= R_thresh]
                    
                    for param in summary.index:
                        if 'jokercondition' not in param:
                            print(f"Setting {param}.")
                            initvalues[param] = summary.loc[param, 'mean']

                        elif 'v_jokercondition[' in param:
                            index = int(param[-4])-2
                            print(f"Setting {param} (index {index}).")
                            initvalues['v_jokercondition'][index] = summary.loc[param, 'mean']

                        elif 'z_jokercondition[' in param:
                            index = int(param[-4])-2
                            print(f"Setting {param} (index {index}).")
                            initvalues['z_jokercondition'][index] = summary.loc[param, 'mean']
                            
                        else:
                            print(f"Not setting {param}.")
                
            elif np.all(summary['r_hat'] <= R_thresh):
                repeat = 0
            
                print("Saving Results.")
                timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                
                if with_lapses:
                    pickle.dump( hssm_model.summary(), 
                                open(f"DDMAngleNolapsesAlljokerswInterceptwLapses/pid{pid}_{timestamp}_Day{day}_ndraws{n_draws}.p", "wb" ) )
                    
                else:
                    pickle.dump( hssm_model.summary(), 
                                open(f"DDMAngleNolapsesAlljokerswIntercept/pid{pid}_{timestamp}_Day{day}_ndraws{n_draws}.p", "wb" ) )                    
            
            print("== == == == ==")
            
