#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:32:09 2024

@author: sascha
"""

from ssms.basic_simulators.simulator import simulator
import numpy as np
import pandas as pd

# Basics
import os
import sys
import time

import arviz as az  # Visualization
import bambi as bmb

# import ssms.basic_simulators # Model simulators
import hddm_wfpt
import hssm
import jax
import pytensor  # Graph-based tensor library
from matplotlib import pyplot as plt

#%%
# Setting float precision in pytensor
pytensor.config.floatX = "float32"
jax.config.update("jax_enable_x64", False)

'''
    Produce dataset
    v: drift rate
    a: boundary separation
    z: starting point
    t: the non-decision time
'''

# Specify parameter values
v_true, a_true, z_true, t_true = [0.5, 1.5, 0.5, 0.2]

# Simulate data
sim_out = simulator([v_true, a_true, z_true, t_true], model="ddm", n_samples=500)

# Turn data into a pandas dataframe
dataset = pd.DataFrame(
    np.column_stack([sim_out["rts"][:, 0], sim_out["choices"][:, 0]]),
    columns=["rt", "response"],
)

dataset


simple_ddm_model = hssm.HSSM(data=dataset)

print(simple_ddm_model)

# simple_ddm_model.graph()

#%%
infer_data_simple_ddm_model = simple_ddm_model.sample(
    sampler="nuts_numpyro",  # type of sampler to choose, 'nuts_numpyro', 'nuts_blackjax' of default pymc nuts sampler
    cores=1,  # how many cores to use
    chains=4,  # how many chains to run
    draws=1000,  # number of draws from the markov chain
    tune=500,  # number of burn-in samples
    idata_kwargs=dict(log_likelihood=True),  # return log likelihood
)  # mp_ctx="forkserver")

az.summary(infer_data_simple_ddm_model)

az.plot_trace(
    infer_data_simple_ddm_model,
    var_names="~log_likelihood",  # we exclude the log_likelihood traces here
)
plt.tight_layout()

az.plot_posterior(simple_ddm_model.traces)

#%%
'''
    With collapsing bounds
    theta: angle of collapsing bound
'''

# Simulate angle data
v_true, a_true, z_true, t_true, theta_true = [0.5, 1.5, 0.5, 0.5, 0.2]
obs_angle = simulator(
    [v_true, a_true, z_true, t_true, theta_true], model="angle", n_samples=1000
)

dataset_angle = pd.DataFrame(
    np.column_stack([obs_angle["rts"][:, 0], obs_angle["choices"][:, 0]]),
    columns=["rt", "response"],
)

model_angle = hssm.HSSM(data=dataset_angle, model="angle")
# model_angle = hssm.HSSM(data=dataset_angle, model="weibull")
model_angle

jax.config.update("jax_enable_x64", False)
infer_data_angle = model_angle.sample(
    sampler="nuts_numpyro",
    chains=2,
    cores=2,
    draws=500,
    tune=500,
    idata_kwargs=dict(log_likelihood=False),  # no need to return likelihoods here
)

az.summary(infer_data_angle)

az.plot_trace(
    infer_data_angle,
    var_names="~log_likelihood",  # we exclude the log_likelihood traces here
)
plt.tight_layout()

az.plot_posterior(model_angle.traces)

#%%

'''
    Hierarchical collapsing bounds DDM
'''

# Make some hierarchical data

n_subjects = 15  # number of subjects
n_trials = 200  # number of trials per subject

sd_v = 0.5  # sd for v-intercept
mean_v = 0.5  # mean for v-intercept

data_list = []
for i in range(n_subjects):
    # Make parameters for subject i
    intercept = np.random.normal(mean_v, sd_v, size=1)
    x = np.random.uniform(-1, 1, size=n_trials)
    y = np.random.uniform(-1, 1, size=n_trials)
    v = intercept + (0.8 * x) + (0.3 * y)

    true_values = np.column_stack(
        [v, np.repeat([[1.5, 0.5, 0.5, 0.1]], axis=0, repeats=n_trials)]
    )

    # Simulate data
    obs_ddm_reg_v = simulator(true_values, model="angle", n_samples=1)

    # Append simulated data to list
    data_list.append(
        pd.DataFrame(
            {
                "rt": obs_ddm_reg_v["rts"].flatten(),
                "response": obs_ddm_reg_v["choices"].flatten(),
                "x": x,
                "y": y,
                "subject": i,
            }
        )
    )

# Make single dataframe out of subject-wise datasets
dataset_reg_v_hier = pd.concat(data_list)
dataset_reg_v_hier

model_reg_v_angle_hier = hssm.HSSM(
    data=dataset_reg_v_hier,
    model="angle",
    include=[
        {
            "name": "v",
            "prior": {
                "Intercept": {
                    "name": "Uniform",
                    "lower": -3.0,
                    "upper": 3.0,
                    "initval": 0.0,
                },
                "x": {"name": "Uniform", 
                      "lower": -1.0, 
                      "upper": 1.0, 
                      "initval": 0.0},
                "y": {"name": "Uniform", 
                      "lower": -1.0, 
                      "upper": 1.0, 
                      "initval": 0.0},
            },
            "formula": "v ~ 1 + (1|subject) + x + y",
            "link": "identity",
        }
    ],
)

jax.config.update("jax_enable_x64", False)
infer_hierarch = model_reg_v_angle_hier.sample(
    sampler="nuts_numpyro", chains=2, cores=1, draws=1000, tune=1000
)

az.summary(infer_hierarch)

az.plot_trace(
    infer_hierarch,
    var_names="~log_likelihood",  # we exclude the log_likelihood traces here
)
plt.tight_layout()

az.plot_posterior(model_reg_v_angle_hier.traces)