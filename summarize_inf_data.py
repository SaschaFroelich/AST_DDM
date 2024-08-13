#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Sun Aug  4 11:03:46 2024
    
    @author: sascha
"""

'''
    1 = random
    2 = congruent
    3 = incongruent
'''

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
import utils
hssm.set_floatX("float32")

# time.sleep(36_000)
import os
import re

initvalues = {'a': 1.0, 
              't': 0.025,
              'theta': 0.9} # a, p, t, v, z

def find_matching_files(directory, regex_pattern):
    """
    List files in a directory that match a regex pattern.

    Args:
    directory (str): Directory path to search files in.
    regex_pattern (str): Regex pattern to match file names.

    Returns:
    list: Paths of files that match the regex pattern.
    """
    # Compile the regex pattern
    pattern = re.compile(regex_pattern)
    
    # List to hold all matching files
    matching_files = []
    
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(directory):
        # Check each file in the directory
        for filename in filenames:
            if pattern.search(filename):  # Using search to find the pattern anywhere in the filename
                # Add full path to the list
                matching_files.append(os.path.join(dirpath, filename))
                
    return matching_files

#%%

if 1:
    files = find_matching_files('/home/sascha/Desktop/vbm_torch/DDM/RLmodel', 
                                'pid')
    
    df = pd.DataFrame(columns=['a',
                                 't',
                                 'theta',
                                 'v_Intercept',
                                 'v_qdiff',
                                 'v_repdiff',
                                 'z_Intercept',
                                 'z_repdiff|jokercondition[1.0]',
                                 'z_repdiff|jokercondition[2.0]',
                                 'z_repdiff|jokercondition[3.0]',
                                 'z_repdiff|jokercondition_mu',
                                 'z_repdiff|jokercondition_sigma'])
    
    
    columns_rename = {'a':'a',
                        't':'t',
                        'theta':'theta',
                        'v_Intercept':'v',
                        'v_qdiff':'v_qdiff',
                        'v_repdiff':'v_repdiff',
                        'z_Intercept':'z',
                        'z_repdiff|jokercondition[1.0]':'z_repdiff1',
                        'z_repdiff|jokercondition[2.0]':'z_repdiff2',
                        'z_repdiff|jokercondition[3.0]':'z_repdiff3',
                        'z_repdiff|jokercondition_mu':'z_repdiff_mu',
                        'z_repdiff|jokercondition_sigma':'z_repdiff_sigma'}
    
    num_columns = 13
    
else:
    files = find_matching_files('/home/sascha/Desktop/vbm_torch/DDM/DDMAngleNolapsesAlljokerswIntercept/Day2', 
                                'pid')
    
    df = pd.DataFrame(columns=['a',
                                 't',
                                 'theta',
                                 'v_Intercept',
                                 'v_jokercondition[2.0]',
                                 'v_jokercondition[3.0]',
                                 'z_Intercept',
                                 'z_jokercondition[2.0]',
                                 'z_jokercondition[3.0]'])

    columns_rename = {'a':'a',
                        't':'t',
                        'theta':'theta',
                        'v_Intercept':'v',
                        'z_Intercept':'z',
                        'v_jokercondition[2.0]':'v_jokercond2',
                        'v_jokercondition[3.0]':'v_jokercond3',
                        'z_Intercept':'z',
                        'z_jokercondition[2.0]':'z_jokercond2',
                        'z_jokercondition[3.0]':'z_jokercond3'}
    
    num_columns = 9

for file in files:
    pid = int(file[-3])
    print(f"\n\nOpening file {file}.")
    summary = pickle.load(open(f"{file}", "rb" ))
    print(summary)
    
    if np.all(summary['r_hat']<1.05):
        dfnew = summary.loc[:, 'mean'].reset_index()
        
        df_pivot = dfnew.pivot_table(index=dfnew.index // len(dfnew['index'].unique()), columns='index', values='mean', aggfunc='sum')
        df_pivot = df_pivot.reset_index(drop=True)
        print(df_pivot)
        
        df = pd.concat((df, df_pivot))
            
                
        
    # if ~np.all(summary['r_hat']<1.02):
    #     'Infer again'
    #     print(f"\n\nInferring again for pid {pid}.")
    #     for key in initvalues.keys():
    #         if summary.loc[key,'r_hat'] < 1.02:
    #             initvalues[key] = summary.loc[key,'r_hat']
        
    #     repeat = 1
    #     while repeat:
    #         n_draws = 1000
    #         n_tune = 1000
             
    #         ddmdata_onesub = ddmdata[ddmdata['participant_id'] == pid]
    #         model_reg_v_angle_hier = hssm.HSSM(
    #             data = ddmdata_onesub,
    #             model = "angle",
    #             hierarchical = False,
    #             categorical = 'jokercondition',
    #             a = bmb.Prior("Uniform", lower=0.1, upper=2.0, initval=initvalues['a']),
    #             t = bmb.Prior("Uniform", lower=0.001, upper=2.0, initval=initvalues['t']),
    #             theta = bmb.Prior("Uniform", lower=0.1, upper=2.0, initval=initvalues['theta']),
                
    #             include=[
    #                 {
    #                     "name": "v",
    #                     "formula": "v ~ 1 + (1|jokercondition)",
    #                     "prior": {
    #                         "1|jokercondition": {"name": "Normal", 
    #                                               # "mu": 0, 
    #                                               "mu": {"name": "Normal", "mu": 2., "sigma": 1., "initval": 2.},
    #                                               "sigma": {"name": "HalfNormal", "sigma": .3, "initval": .1},
    #                                               "initval": 2},
    #                     },
    #                     "link": "identity",
    #                 },
    #                 {
    #                     "name": "z",
    #                     "formula": "z ~ 1 + (1|jokercondition)",
    #                     "prior": {
    #                         "Intercept": {
    #                             "name": "Uniform",
    #                             "lower": 0.3,
    #                             "upper": 0.7,
    #                             "initval": 0.5},
    #                     },
    #                     "link": "identity",
    #                 }
    #             ],
    #         )
            
            
    #         infer_data_reg_v_a = model_reg_v_angle_hier.sample(
    #             sampler="nuts_numpyro", chains=4, cores=4, draws=n_draws, tune=n_tune,
    #             include_mean = False
    #             )
            
    #         if np.all(az.summary(infer_data_reg_v_a)[0:13]['r_hat']<1.02):
    #             repeat = 0

    #         else:
    #             n_draws += 1000
    #             n_tune += 1000
        
    #     print(ddmdata_onesub['participant_id'].unique())
    #     print(az.summary(infer_data_reg_v_a)[0:14])
    #     timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    #     pickle.dump( az.summary(infer_data_reg_v_a)[0:14], 
    #                 open(f"{file}", "wb" ))

#%%

import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(num_columns, figsize = (8, np.floor(num_columns*1.5)))

df.rename(columns=columns_rename, inplace = True)

colidx = 0
for col in df.columns:
    print(col)
    if col != 'v[0]' and col != 'v[1]' and col != 'v[2]' and col != 'v[3]' and col != 'v[4]':
        print("\n")
        print(f"Variable {col}.")
        print(f"mean: {df[col].mean()}.")
        print(f"stdev: {df[col].std()}.")
        
        
        sns.kdeplot(df[col], ax=axes[colidx])
        axes[colidx].set_ylabel(col)
        colidx += 1
plt.savefig('res_day1.png')
plt.show()

#%%

utils.cohens_d(df['v_jokercondition[2.0]'],df['v_jokercondition[3.0]'])
utils.cohens_d(df['z_jokercondition[2.0]'],df['z_jokercondition[3.0]'])
