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
'''

# ddmdata = ddmdata[ddmdata['jokercondition'] == 2]
# time.sleep(36_000)

R_thresh = 1.05

for day in range(1, 2):
    ddmdata = DDMutils.get_DDM_data2(day)
    ddmdata = ddmdata[ddmdata['participant_id']<=10]
    
    ddmdata.rename(columns={'subject': 'participant_id'}, inplace = True)
    ddmdata['jokercondition'] = ddmdata['jokercondition'].apply(lambda x: 'A' if x == 1 else 'B' if x == 2 else 'C' if x == 3 else None)
    
    n_draws = 6_0
    n_tune = 6_0
    repeat = 1
    num_rep = 1
    
    initvalues = {'a_Intercept': 0.8, 
                  'a_1|participant_id_mu': 0,
                  't_Intercept': 0.2, 
                  't_1|participant_id_mu': 0,
                  'theta_Intercept': 0.9, 
                  'theta_1|participant_id_mu': 0,
                  'z_Intercept': 0.5,
                  'z_jokercondition': [0., 0.],
                  'z_jokercondition|participant_id_mu': [0., 0.],
                  'v_Intercept': 2.0,
                  'v_jokercondition': [0., 0.]} # a, p, t, v, z
    
    while repeat and num_rep < 4:
        model_reg_v_angle_hier = hssm.HSSM(
            data = ddmdata,
            model = "angle",
            hierarchical = True,
            categorical = 'jokercondition',
            include=[
                {
                    "name": "v",
                    "formula": "v ~ 1 + jokercondition + (1 + jokercondition|participant_id)",
                    "prior": {
                        # "jokercondition": {"name": "Normal", 
                        #                       "mu": {"name": "Normal", "mu": 2., "sigma": 1.},
                        #                       "sigma": {"name": "HalfNormal", "sigma": .3}},
                        "jokercondition|participant_id": {"name": "Normal", 
                                              "mu": {"name": "Normal", "mu": 2., "sigma": 1.},
                                              "sigma": {"name": "HalfNormal", "sigma": .3}},
                        "Intercept": {"name": "Normal", "mu": 1, "sigma": 2, "initval": 1},
                    },
                    "link": "identity",
                },
                {
                    "name": "z",
                    "formula": "z ~ 1 + jokercondition + (1 + jokercondition|participant_id)",
                    "prior": {
                        "Intercept": {
                            "name": "Uniform",
                            "lower": 0.3,
                            "upper": 0.7,
                            "initval": 0.5},
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
                            "initval": 0.8},
                        "1|participant_id": {"name": "Normal", 
                                              "mu": {"name": "Normal", "mu": 0., "sigma": 1.},
                                              "sigma": {"name": "HalfNormal", "sigma": .3}},
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
                            "initval": 0.25},
                        "1|participant_id": {"name": "Normal", 
                                              "mu": {"name": "Normal", "mu": 0, "sigma": 0.5},
                                              "sigma": {"name": "HalfNormal", "sigma": .3}},
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
                            "initval": 0.9},
                        "1|participant_id": {"name": "Normal", 
                                              "mu": {"name": "Normal", "mu": 0, "sigma": 0.5},
                                              "sigma": {"name": "HalfNormal", "sigma": .3}},
                    },
                    "link": "identity",
                },
                
                # {
                #     "name": "t",
                #     "formula": "t ~  1 + (1|participant_id)",
                #     "prior": {
                #         # "Intercept": {
                #         #     "name": "Uniform",
                #         #     "lower": 0.3,
                #         #     "upper": 0.7,
                #         #     "initval": 0.5},
                #         # "1|participant_id": {
                #         #     "name": "Uniform",
                #         #     "lower": 0.3,
                #         #     "upper": 0.7,
                #         #     "initval": 0.5},
                #     },
                #     "link": "identity",
                # },
                # {
                #     "name": "theta",
                #     "formula": "theta ~  1 + (1|participant_id)",
                #     "prior": {
                #         # "Intercept": {
                #         #     "name": "Uniform",
                #         #     "lower": 0.3,
                #         #     "upper": 0.7,
                #         #     "initval": 0.5},
                #         # "1|participant_id": {
                #         #     "name": "Uniform",
                #         #     "lower": 0.3,
                #         #     "upper": 0.7,
                #         #     "initval": 0.5},
                #     },
                #     "link": "identity",
                # },
            ],
            )
            
        
        infer_data_trace = model_reg_v_angle_hier.sample(
            sampler="nuts_numpyro", chains=4, cores=4, draws=n_draws, tune=n_tune,
            include_mean = False
            )
        
        if np.any(az.summary(infer_data_trace, var_names = ['a_Intercept',
                                                  'a_1|participant_id_mu',
                                                  't_Intercept',
                                                  't_1|participant_id_mu',
                                                  'theta_Intercept',
                                                  'theta_1|participant_id_mu',
                                                  'v_Intercept', 
                                                  'v_jokercondition',
                                                  'v_jokercondition|participant_id_mu',
                                                  'z_Intercept',
                                                  'z_jokercondition',
                                                  'z_jokercondition|participant_id_mu'])['r_hat']>R_thresh):
            
            n_draws += 1000
            n_tune += 1000
            num_rep += 1
            
            '''
                Set initial values of those parameters with rhat < 1.05
            '''
            if np.any(az.summary(infer_data_trace, var_names = ['a_Intercept',
                                                      'a_1|participant_id_mu',
                                                      't_Intercept',
                                                      't_1|participant_id_mu',
                                                      'theta_Intercept',
                                                      'theta_1|participant_id_mu',
                                                      'v_Intercept', 
                                                      'v_jokercondition',
                                                      'v_jokercondition|participant_id_mu',
                                                      'z_Intercept',
                                                      'z_jokercondition',
                                                      'z_jokercondition|participant_id_mu'])['r_hat']<=R_thresh):
                
                
                
                pass
    
            
            dfgh
            
        else:
            repeat = 0

            print(f"Saving Results for pid {pid}.")
            summary = az.summary(infer_data_trace, var_names = ['a_Intercept',
                                                      'a_1|participant_id_mu',
                                                      'a_1|participant_id_sigma',
                                                      't_Intercept',
                                                      't_1|participant_id_mu',
                                                      't_1|participant_id_sigma',
                                                      'theta_Intercept',
                                                      'theta_1|participant_id_mu',
                                                      'theta_1|participant_id_sigma',
                                                      'v_Intercept', 
                                                      'v_jokercondition',
                                                      'v_jokercondition|participant_id_mu',
                                                      'v_jokercondition|participant_id_sigma',
                                                      'z_Intercept',
                                                      'z_jokercondition',
                                                      'z_jokercondition|participant_id_mu',
                                                      'z_jokercondition|participant_id_sigma'])
            
            
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
    
#%%

az.summary(infer_data_trace, var_names = ['a_Intercept',
                                          'a_1|participant_id_mu',
                                          't_Intercept',
                                          't_1|participant_id_mu',
                                          'theta_Intercept',
                                          'theta_1|participant_id_mu',
                                          'v_Intercept', 
                                          'v_jokercondition',
                                          'v_jokercondition|participant_id_mu',
                                          'z_Intercept',
                                          'z_jokercondition',
                                          'z_jokercondition|participant_id_mu'])['r_hat']