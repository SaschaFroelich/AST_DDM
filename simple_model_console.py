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
with_lapses = False

for day in range(1, 2):
    
    ddmdata = DDMutils.get_DDM_data2(day)
    
    initvalues = {'a': 0.8, 
                  't': 0.05,
                  'theta': 0.2, 
                  'z_Intercept': 0.5,
                  'v_Intercept': 2.0,
                  'v_jokercondition': [0, 0],
                  'z_jokercondition': [0, 0]}
    
    ddmdata.rename(columns={'subject': 'participant_id'}, inplace = True)
    
    for pid in [18, 20, 26, 31, 34, 36, 38, 41, 43, 45, 48, 49, 51, 58, 60]:
        n_draws = 2_0
        n_tune = 2_0
        repeat = 1
        num_rep = 1
        
        while repeat and num_rep < 4:        
            print(f"Inference for pid {pid} on day {day}. Repetition number {num_rep}.")    
            ddmdata_onesub = ddmdata[ddmdata['participant_id'] == pid]

            if with_lapses:
                model_reg_v_angle = hssm.HSSM(
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
                model_reg_v_angle = hssm.HSSM(
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
                
            infer_data_trace = model_reg_v_angle.sample(
                sampler="nuts_numpyro", chains=4, cores=4, draws=n_draws, tune=n_tune,
                include_mean = False
                )
            
            summary = model_reg_v_angle.summary()
            print(summary)
            
            if np.any(summary['r_hat'] > R_thresh):
                n_draws += 2000
                n_tune += 2000
                num_rep += 1
                
                if np.any(summary['r_hat'] <= R_thresh):
                    summary = summary[summary['r_hat'] <= R_thresh]
                    
                    for param in summary.index:
                        if 'jokercondition' not in param:
                            print(f"Setting {param}.")
                            initvalues[param] = summary.loc[param, 'mean']

                        elif 'v_jokercondition[' in param:
                            index = int(param[-4])-1
                            print(f"Setting {param} (index {index}).")
                            initvalues['v_jokercondition'][index] = summary.loc[param, 'mean']

                        elif 'z_jokercondition[' in param:
                            index = int(param[-4])-1
                            print(f"Setting {param} (index {index}).")
                            initvalues['z_jokercondition'][index] = summary.loc[param, 'mean']
                
            else:
                repeat = 0
            
                print("Saving Results.")
                timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                
                if with_lapses:
                    pickle.dump( model_reg_v_angle.summary(), 
                                open(f"DDMAngleNolapsesAlljokerswInterceptwLapses/pid{pid}_{timestamp}_Day{day}.p", "wb" ) )
                    
                else:
                    pickle.dump( model_reg_v_angle.summary(), 
                                open(f"DDMAngleNolapsesAlljokerswIntercept/pid{pid}_{timestamp}_Day{day}.p", "wb" ) )                    
            
            print("== == == == ==")
            
