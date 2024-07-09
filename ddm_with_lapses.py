#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:59:24 2024

@author: sascha
"""

import numpy as np
import matplotlib as plt
import hssm
import tqdm as notebook_tqdm
import os
import pandas as pd
import arviz as az
import bambi as bmb
from datetime import datetime
import pickle
import DDMutils

hssm.set_floatX("float32")
ddmdata = DDMutils.get_DDM_data(day=2)

#%%
'''
    Loop over individual subjects.
    Regular DDM with lapses
'''

import warnings
results_df = pd.DataFrame()
# results_df_new = pd.DataFrame({"subject": [],
#                                "a": [],
#                                "t": [],
#                                "theta": [],
#                                "v_1|jokercondition": [],
#                                "v_1|jokercondition_sigma": [],
#                                "v_Intercept": [],
#                                "z_1|jokercondition": [],
#                                "z_1|jokercondition_sigma": [],
#                                "z_Intercept": [],
#                                "v": [],
#                                "z": [],
#                                "num_samples": []})

# Suppress all warnings
warnings.filterwarnings("ignore")

n_draws_max = 2000
n_tune_max = 2000

for i in ddmdata['subject'].unique():
    repeat = 1
    n_draws = 2000
    n_tune = 2000
    
    print(f"\n\n===== Doing subject {i} ======")
    while repeat:
        ddmdata_model = ddmdata[ddmdata['subject'] == i]
        
        print(f"Fitting for subjects {ddmdata['subject'].unique()}")
        model_reg_v_angle_hier = hssm.HSSM(
            data = ddmdata_model,
            model = "ddm",
            p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.5},
            lapse=bmb.Prior("Uniform", lower=0.0, upper=20.0),
            include = [
                {
                    "name": "v",
                    "prior": {
                        "Intercept": {
                            "name": "Uniform",
                            "lower": -4.0,
                            "upper": 4.0,
                            "initval": 2.0,
                        },
                    },
                    "formula": "v ~ 1 + (1|jokercondition)",
                    "link": "identity",
                    
                }, 
                {
                      "name": "z",
                      "prior": { 
                          "Intercept": {
                              "name": "Uniform",
                              "lower": 0.1,
                              "upper": 0.9,
                              "initval": 0.5,
                          },
                      },
                      "formula": "z ~ 1 + (1|jokercondition)",
                      "link": "identity",
                  },
            ],
            )

        infer_data_reg_v_a = model_reg_v_angle_hier.sample(
            sampler="nuts_numpyro", 
            chains=4, 
            cores=4, 
            draws=n_draws, 
            tune=n_tune # number of burn-in samples
        )
        
        print(az.summary(infer_data_reg_v_a))
        
        if (az.summary(infer_data_reg_v_a)['r_hat'] > 1.02).any():
            n_draws += 1000
            n_tune += 1000
            print(f"\n\n===== REPEATING subject {i} ======")
            repeat = 1
            
        else:
            if n_draws > n_draws_max: n_draws_max = n_draws
            if n_tune > n_tune_max: n_tune_max = n_tune
            repeat = 0
            
            summary_df = az.summary(infer_data_reg_v_a).iloc[0:13, :]
            summary_df['ag_idx'] = i
            summary_df['num_samples'] = n_draws + n_tune
            results_df = pd.concat((results_df, summary_df))
            # df_subject = pd.DataFrame({
            # "subject": [i],
            # "a": [infer_data_reg_v_a.posterior['a'].mean()],
            # "t": [infer_data_reg_v_a.posterior['t'].mean()],
            # "theta": [infer_data_reg_v_a.posterior['theta'].mean()],
            # "v_1|jokercondition": [infer_data_reg_v_a.posterior['v_1|jokercondition'].mean(axis=0).mean(axis=0)],
            # "v_1|jokercondition_sigma": [infer_data_reg_v_a.posterior['v_1|jokercondition_sigma'].mean()],
            # "v_Intercept": [infer_data_reg_v_a.posterior['v_Intercept'].mean()],
            # "z_1|jokercondition": [infer_data_reg_v_a.posterior['z_1|jokercondition'].mean(axis=0).mean(axis=0)],
            # "z_1|jokercondition_sigma": [infer_data_reg_v_a.posterior['z_1|jokercondition_sigma'].mean()],
            # "z_Intercept": [infer_data_reg_v_a.posterior['z_Intercept'].mean()],
            # "z": [],
            # "num_samples": []})

timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

pickle.dump( results_df, 
            open(f"ddm_lapses_flat_jokercond_{timestamp}.p", "wb" ) )
print(f"n_draws_max is {n_draws_max}.")
