#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 11:21:30 2024

@author: sascha
"""

import pandas as pd
import numpy as np

def get_DDM_data(day):
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
    # ddmdata = pd.DataFrame(
    #     np.column_stack([
    #         data_day1["PB"],
    #         data_day1["RT"],
    #         data_day1["response"],
    #         data_day1["qdiff"],
    #         data_day1["repdiff"],
    #         data_day1['jokercondition']
    #     ]),  # Make sure this closing bracket matches with np.column_stack opening
    #     columns=["subject", "rt", "response", "qdiff", "repdiff", "jokercondition"]
    # )          

    if day == 2:
        ddmdata = pd.DataFrame({'subject': data_day2['PB'],
                                'rt': data_day2['RT'],
                                'response': data_day2['response'],
                                'jokercondition': data_day2['jokercondition']})       
        

    ddmdata['jokercondition'] = ddmdata['jokercondition'].map(lambda x: 'R' if x == 1 else 
                                                              'C' if x == 2 else 
                                                              'I' if x== 3 else None)
    
    assert None not in ddmdata['jokercondition']

    ddmdata['subject'] = ddmdata['subject'].astype(int)
    ddmdata['subject'] = ddmdata['subject'] - 1
    return ddmdata