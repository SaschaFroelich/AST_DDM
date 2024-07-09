#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:00:23 2024

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
        [v, np.repeat([[1.5, 0.5, 0.5, 0.0]], axis=0, repeats=n_trials)]
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
                "x": {"name": "Uniform", "lower": -1.0, "upper": 1.0, "initval": 0.0},
                "y": {"name": "Uniform", "lower": -1.0, "upper": 1.0, "initval": 0.0},
            },
            "formula": "v ~ 1 + (1|subject) + x + y",
            "link": "identity",
        }
    ],
)

jax.config.update("jax_enable_x64", False)
out = model_reg_v_angle_hier.sample(
    sampler="nuts_numpyro", chains=2, cores=1, draws=30, tune=30
)