#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:49:56 2024

@author: sascha
"""

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
import numpyro
hssm.set_floatX("float64")

numpyro.set_host_device_count(4)

'''
    Individual participants, all jokertypes
    With v ~ 1 + jokercondition
    i.e., not hierarchically modelled
'''

# ddmdata = ddmdata[ddmdata['jokercondition'] == 2]

# time.sleep(36_000)


R_thresh = 1.05

for day in range(1, 2):
    
    ddmdata = DDMutils.get_DDM_data2(day)
    
    initvalues = {'a': 0.8, 
                  't': 0.05,
                  'theta': 0.2, 
                  'z': 0.5,
                  'v': 2.0} # a, p, t, v, z
    
    ddmdata.rename(columns={'subject': 'participant_id'}, inplace = True)
    
    for pid in range(21, 60):
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
                # p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.5},
                # lapse=bmb.Prior("Uniform", lower=0.0, upper=20.0),
                a = bmb.Prior("Uniform", lower=0.1, upper=2.0, initval=1.0),
                t = bmb.Prior("Uniform", lower=0.001, upper=2.0, initval=0.025),
                # z = bmb.Prior("Uniform", lower=0.1, upper=0.9, initval=0.5),
                # v = bmb.Prior("Uniform", lower=0.1, upper=6.0, initval=2),
                theta = bmb.Prior("Uniform", lower=0.1, upper=2.0, initval=0.9),
                include=[
                    {
                        "name": "v",
                        # "formula": "v ~ 1 + (1|jokercondition)",
                        "formula": "v ~ 1 + jokercondition",
                        "prior": {
                            # "1|jokercondition": {"name": "Normal", 
                            #                       # "mu": 0, 
                            #                       "mu": {"name": "Normal", "mu": 2., "sigma": 1., "initval": 2.},
                            #                       "sigma": {"name": "HalfNormal", "sigma": .3, "initval": .1},
                            #                       "initval": 2},
                            "Intercept": {"name": "Normal", "mu": 1, "sigma": 2, "initval": 1},
                            # "jokercondition": {"name": "Normal", 
                            #                    "mu": 0, 
                            #                    "sigma": 1,
                            #                    "initval": [0., 0., 0.]},
                        },
                        "link": "identity",
                    },
                    {
                        "name": "z",
                        # "formula": "z ~ 1 + (1|jokercondition)",
                        "formula": "z ~ 1 + jokercondition",
                        "prior": {
                            "Intercept": {
                                "name": "Uniform",
                                "lower": 0.3,
                                "upper": 0.7,
                                "initval": 0.5},
                            # "jokercondition": {"name": "Uniform", 
                            #                     "lower": -0.2,
                            #                     "upper": 0.2,
                            #                     "initval": [0., 0., 0.]},
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