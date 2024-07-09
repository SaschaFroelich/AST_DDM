#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:51:09 2024

@author: benwagner
"""

#%%
import numpy as np
import matplotlib as plt
import hssm
import tqdm as notebook_tqdm
import os
import pandas as pd

#%%
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

#%%
hssm.set_floatX("float32")

#%%
#simulate a dataset

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
#%%

#create q-value diff
random_variable = np.random.uniform(low=0.4, high=0.6, size= len(data))
data['qdiff'] = random_variable

#%%
#create distinct dataframes for day1 and 2

data_day1 = data[data['day'] == 1]
data_day2 = data[data['day'] == 2]


#%%
#save variables for ddm in ddmdata
# Example data setup
#ddmdata = {
#    'subject_id': data_day1['PB'],
#    'rt': data_day1['RT'],  # reaction times
#   'response': data_day1['choice_GD']
#}

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


#%%

model_reg_v_angle_hier = hssm.HSSM(
    data=ddmdata,
    model="angle",
    include=[
        {
            "name": "v",
            "prior": {
                "Intercept": {
                    "name": "Uniform",
                    "lower": -4.0,
                    "upper": 4.0,
                    "initval": 2.0,
                },
                "qdiff": {"name": "Uniform", "lower": -2.0, "upper": 2.0, "initval": 0.0},
                "repdiff": {"name": "Uniform", "lower": -2.0, "upper": 2.0, "initval": 0.0},
                # Additional priors for interaction terms
                "qdiff:jokercondition": {"name": "Uniform", "lower": -2.0, "upper": 2.0, "initval": 0.0},
                "repdiff:jokercondition": {"name": "Uniform", "lower": -2.0, "upper": 2.0, "initval": 0.0},
            },
            "formula": "v ~ 1 + qdiff + repdiff + qdiff:C(jokercondition) + repdiff:C(jokercondition) + 1 + qdiff + repdiff|subject",
            "link": "identity",
        }
    ],
)


#%%
#model_reg_v_angle_hier.graph()

#%%
infer_data_reg_v_a = model_reg_v_angle_hier.sample(
    sampler="nuts_numpyro", chains=4, cores=4, draws=30, tune=30
)

