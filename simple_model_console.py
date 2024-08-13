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

for day in range(2, 3):
    
    ddmdata = DDMutils.get_DDM_data2(day)
    
    initvalues = {'a': 0.8, 
                  't': 0.05,
                  'theta': 0.2, 
                  'z': 0.5,
                  'v': 2.0} # a, p, t, v, z
    
    ddmdata.rename(columns={'subject': 'participant_id'}, inplace = True)
    
    for pid in range(34, 60):
        n_draws = 1000
        n_tune = 1000
        repeat = 1
        num_rep = 1
        
        while repeat and num_rep < 4:        
            print(f"Inference for pid {pid} on day {day}. Repetition number {num_rep}.")    
            ddmdata_onesub = ddmdata[ddmdata['participant_id'] == pid]
            model_reg_v_angle_hier = hssm.HSSM(
                data = ddmdata_onesub,
                model = "angle",
                hierarchical = False,
                categorical = 'jokercondition',
                include=[
                    {
                        "name": "v",
                        "formula": "v ~ 1 + jokercondition",
                        "prior": {
                            "Intercept": {"name": "Normal", "mu": 1, "sigma": 2, "initval": 1},
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
                                "initval": 0.5},
                        },
                        "link": "identity",
                    }
                ],
            )
            
            
            infer_data_reg_v_a = model_reg_v_angle_hier.sample(
                sampler="nuts_numpyro", chains=4, cores=4, draws=n_draws, tune=n_tune,
                include_mean = False
                )
            
            print(ddmdata_onesub['participant_id'].unique())
            print(az.summary(infer_data_reg_v_a)[0:10])
            
            if ~np.all(az.summary(infer_data_reg_v_a)[0:10]['r_hat']<R_thresh):
                n_draws += 1000
                n_tune += 1000
                num_rep += 1
                
            else:
                repeat = 0
            
                print("Saving Results.")
                timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                pickle.dump( az.summary(infer_data_reg_v_a)[0:14], 
                            open(f"DDMAngleNolapsesAlljokerswIntercept/{timestamp}_Day{day}_pid{pid}.p", "wb" ) )
            
            print("== == == == ==")
            
