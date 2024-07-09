#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:52:28 2024

@author: benwagner
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:51:09 2024

@author: benwagner
"""

import numpy as np
import matplotlib as plt
import hssm
import tqdm as notebook_tqdm
import os
import pandas as pd
import arviz as az
import bambi as bmb
hssm.set_floatX("float32")

#%%
# Assuming the working directory is set to where the file is located
filename = 'ddm_data_gefiltert_choiceonly.csv'
data = pd.read_csv(filename)

# Check the first few rows of the data
print(data.head())

#Jokertype 1 = random, 2 = congruent, 3 incongruent
data['jokercondition'] = data['jokertypes'] +1
data['jokercondition'] = pd.Categorical(data['jokercondition'], ordered=True, categories=[1, 2, 3])

#data["jokercondition"] = data["jokercondition"].map({1: "random", 2: "congruent",3: "incongruent"})
#recode response column
data['response'] = data['choices_GD'].replace(0,-1)

'''
    Create random q-value diff
'''
random_variable = np.random.uniform(low=0.4, high=0.6, size= len(data))
data['qdiff'] = random_variable

'''
    Create distinct dataframes for day1 and 2
'''
data_day1 = data[data['day'] == 1]
data_day2 = data[data['day'] == 2]

'''
    Save variables for ddm in ddmdata
'''

ddmdata = pd.DataFrame(
    np.column_stack([
        data_day1["PB"],
        data_day1["RT"],
        data_day1["response"],
        data_day1["qdiff"],
        data_day1["repdiff"],
        data_day1['jokercondition']
    ]),  # Make sure this closing bracket matches with np.column_stack opening
    columns=["subject", "rt", "response", "qdiff", "repdiff", "jokercondition"]
)                         

ddmdata['jokercondition'] = ddmdata['jokercondition'].map(lambda x: 'R' if x == 1 else 'C' if x == 2 else 'I' if x== 3 else None)

ddmdata['subject'] = ddmdata['subject'].astype(int)
ddmdata['subject'] = ddmdata['subject'] - 1
# ddmdata['response'] = ddmdata['response'].astype(int)

#%%

ddmdata = ddmdata.drop('jokercondition', axis = 1)
ddmdata = ddmdata.drop('qdiff', axis = 1)
ddmdata = ddmdata.drop('repdiff', axis = 1)

# ddmdata_model= ddmdata[ddmdata['subject'] == 2]

ddmdata_model = ddmdata.copy()

print(f"Fitting for subjects {ddmdata['subject'].unique()}")
model_reg_v_angle_hier = hssm.HSSM(
    data = ddmdata_model,
    model = "ddm",
    p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.5},
    lapse=bmb.Prior("Uniform", lower=0.0, upper=20.0)
    # include = [
    #     {
    #         "name": "v",
    #         "prior": {
    #             "Intercept": {
    #                 "name": "Uniform",
    #                 "lower": -4.0,
    #                 "upper": 4.0,
    #                 "initval": 2.0,
    #             },
    #             "qdiff": {"name": "Uniform", 
    #                       "lower": -2.0, 
    #                       "upper": 2.0, 
    #                       "initval": 0.0},
    #             "repdiff": {"name": "Uniform", 
    #                         "lower": -2.0, 
    #                         "upper": 2.0, 
    #                         "initval": 0.0},
    #             # Additional priors for interaction terms
    #             "qdiff:jokercondition": {"name": "Uniform", 
    #                                       "lower": -2.0, 
    #                                       "upper": 2.0, 
    #                                       "initval": 0.0},
    #             "repdiff:jokercondition": {"name": "Uniform", 
    #                                         "lower": -2.0, 
    #                                         "upper": 2.0, 
    #                                         "initval": 0.0},
    #         },
    #         "formula": "v ~ 1 + (1|jokercondition)",
    #         "link": "identity",
            
    #     }, 
    #     {
    #           "name": "z",
    #           "prior": {
    #               "Intercept": {
    #                   "name": "Uniform",
    #                   "lower": 0.1,
    #                   "upper": 0.9,
    #                   "initval": 0.5,
    #               },
    #               "qdiff": {"name": "Uniform", "lower": -0.1, "upper": 0.1, "initval": 0.0},
    #               "repdiff": {"name": "Uniform", "lower": -0.1, "upper": 0.1, "initval": 0.0},
    #           },
    #            "formula": "z ~ 1 + (1|jokercondition)",
    #           "link": "identity",
    #       },
    # ],
)


#%%
#model_reg_v_angle_hier.graph()

#%%
infer_data_reg_v_a = model_reg_v_angle_hier.sample(
    sampler="nuts_numpyro", 
    chains=4, 
    cores=4, 
    draws=1_000, 
    tune=1_000 # number of burn-in samples
)


#%%

import pickle
pickle.dump( infer_data_reg_v_a,  open(f"DDM_fit_DDMLapse.p", "wb" ) )

if 0:
    az.plot_trace(infer_data_reg_v_a, var_names=["a","t","v","z","theta"], combined=True)
    
    az.plot_trace(infer_data_reg_v_a, var_names=["a","t","v","z"], combined=True)
    
    az.summary(infer_data_reg_v_a)
    
    rhat = az.rhat(infer_data_reg_v_a.posterior[['v_Intercept', 'v_1|jokercondition']])
    

#%%

'''
    Loop over individual subjects.
    DDM no angle, with lapses
'''

import warnings

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
            lapse=bmb.Prior("Uniform", lower=0.0, upper=20.0)
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
            

            
            
#%%

'''
    Loop over individual subjects.
    Angle with lapses
'''

import warnings

ddmdata_model = ddmdata

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
            model = "angle",
            p_outlier={"name": "Uniform", "lower": 0.0001, "upper": 0.5},
            lapse=bmb.Prior("Uniform", lower=0.0, upper=20.0)
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