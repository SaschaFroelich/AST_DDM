#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 11:21:30 2024

@author: sascha
"""
import pandas as pd
import numpy as np

def find_resp_options(stimulus_mat):
    '''
    Given a dual-target stimulus (e.g. 12, 1-indexed), this function returns the two response
    options in 0-indexing. E.g.: stimulus_mat=14 -> option1_python = 0, option1_python = 3.
    Returns twice the same option in case of STT.

    Parameters
    ----------
    stimulus_mat : torch tensor with shape [num_agents]
        DESCRIPTION.

    Returns
    -------
    option1_python : tensor with shape [num_agents]
        -1 in case of single-target trial.
        Otherwise option 1 of dual-target trial. 0-indexed.
    
    option2_python : tensor with shape [num_agents]
        -1 in case of single-target trial.
        Otherwise option 1 of dual-target trial. 0-indexed.

    '''
    
    assert isinstance(stimulus_mat, int)
    
    # # assert(torch.is_tensor(stimulus_mat))
    option2_python = int(((stimulus_mat % 10) - 1))
    option1_python = int((((stimulus_mat - (stimulus_mat % 10)) / 10) -1))
    
    "Make sure to return option2 twice in case of STT"
    # option1_python = np.where(stimulus_mat > 10, option1_python, option2_python)
    
    return option1_python, option2_python

def return_qdiff(row):
    '''
        qpdiff = Q(optimal_response) - Q(suboptimal_response)
    '''

    opt1, opt2 = find_resp_options(row['trialsequence'])
    
    if row['group'] <= 1:
        optimal_responses = [0, 3]
        
    elif row['group'] > 1:
        optimal_responses = [1, 2]
    
    assert int(opt1 in optimal_responses) + int(opt2 in optimal_responses) == 1
    
    # print(f"{row['trialsequence']} -> {opt1}, {opt2}.")

    Q1 = row['Qs'][opt1]
    Q2 = row['Qs'][opt2]
    
    if opt1 in optimal_responses:
        res = Q1 - Q2
    
    elif opt2 in optimal_responses:
        res = Q2 - Q1
        
    else:
        raise Exception("Nope.")
    
    # if row['ID'] == '5d7ebf9e93902b0001965912' and row['trialidx'] == 19:
    #     print(f"returning {res}.")
        
    return res

def return_repdiff(row):
    '''
        repdiff = rep(optimal_response) - rep(suboptimal_response)
    '''

    opt1, opt2 = find_resp_options(row['trialsequence'])
    
    if row['group'] <= 1:
        optimal_responses = [0, 3]
        
    elif row['group'] > 1:
        optimal_responses = [1, 2]
    
    assert int(opt1 in optimal_responses) + int(opt2 in optimal_responses) == 1

    # print(f"{row['trialsequence']} -> {opt1}, {opt2}.")

    R1 = row['repvals'][opt1]
    R2 = row['repvals'][opt2]
    
    if opt1 in optimal_responses:
        res = R1 - R2
    
    elif opt2 in optimal_responses:
        res = R2 - R1
    
    return res
    
# def get_DDM_data(day):
#     # Assuming the working directory is set to where the file is located
#     filename = 'ddm_data_gefiltert_choiceonly.csv'
#     data = pd.read_csv(filename)

#     # Check the first few rows of the data
#     print(data.head())

#     # Jokertype 1 = random, 2 = congruent, 3 incongruent
#     data['jokercondition'] = data['jokertypes'] + 1
#     data['jokercondition'] = pd.Categorical(data['jokercondition'], ordered=True, categories=[1, 2, 3])

#     # Recode response column
#     data['response'] = data['choices_GD'].replace(0, -1)

#     '''
#         Create random q-value diff
#     '''
#     random_variable = np.random.uniform(low=0.4, high=0.6, size= len(data))
#     data['qdiff'] = random_variable

#     '''
#         Create distinct dataframes for day1 and 2
#     '''
#     data_day1 = data[data['day'] == 1]
#     data_day2 = data[data['day'] == 2]

#     if day == 2:
#         ddmdata = pd.DataFrame({'subject': data_day2['PB'],
#                                 'rt': data_day2['RT'],
#                                 'response': data_day2['response'],
#                                 'jokercondition': data_day2['jokercondition']})       
        
#     elif day == 1: 
#         ddmdata = pd.DataFrame({'subject': data_day1['PB'],
#                                 'rt': data_day1['RT'],
#                                 'response': data_day1['response'],
#                                 'jokercondition': data_day1['jokercondition']})       
        

#     # ddmdata['jokercondition'] = ddmdata['jokercondition'].map(lambda x: 'R' if x == 1 else 
#     #                                                           'C' if x == 2 else 
#     #                                                           'I' if x== 3 else None)
    
#     assert None not in ddmdata['jokercondition']

#     # ddmdata['subject'] = ddmdata['subject'].astype("object")
#     # ddmdata['subject'] = ddmdata['subject'] - 1
#     return ddmdata

def get_DDM_data2(day, RL = False):
    import ast
    # filename = 'ddm_data_gefiltert_choiceonly.csv'
    # data = pd.read_csv(filename)

    data = pd.read_csv('Data_DDM_2024-08-08 14:27:50.csv')
    data = data[data['trialsequence']>10]
    
    data['Qs']  = data[['Q1', 'Q2', 'Q3', 'Q4']].apply(lambda row: list(row), axis=1)
    
    print("RT to sec")
    data['RT'] /= 1_000
    
    print("Filtering out data < 150ms.")
    data = data[data['RT'] > 0.15]
    
    print("Filtering out erroneous responses.")
    data = data[data['choices'] != -2]
    
    # Check the first few rows of the data
    print(data.head())
    
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], axis=1, inplace = True)

    #Jokertype 1 = random, 2 = congruent, 3 incongruent
    data['jokercondition'] = data['jokertypes'] + 1
    data['jokercondition'] = pd.Categorical(data['jokercondition'], ordered=True, categories=[1,2,3])

    #data["jokercondition"] = data["jokercondition"].map({1: "random", 2: "congruent",3: "incongruent"})
    #recode response column
    data['response'] = data['choices_GD'].replace(0,-1)
    
    data['repvals'] = data['repvals'].apply(ast.literal_eval)

    # Create q-value diff
    data['qdiff'] = data.apply(lambda row: return_qdiff(row), axis = 1)
    data['repdiff'] = data.apply(lambda row: return_repdiff(row), axis = 1)
    
    #create distinct dataframes for day1 and 2
    data_day1 = data[data['day'] == 1]
    data_day2 = data[data['day'] == 2]
    
    if day == 1:
        ddmdata = pd.DataFrame(
            np.column_stack([
                data_day1["ag_idx"],
                data_day1["RT"],
                data_day1["response"],
                data_day1['jokercondition'],
                data_day1['qdiff'],
                data_day1['repdiff']
            ]),  # Make sure this closing bracket matches with np.column_stack opening
            columns=["participant_id", "rt", "response", "jokercondition", "qdiff", "repdiff"]
        )     
    
    elif day == 2:
        ddmdata = pd.DataFrame(
            np.column_stack([
                data_day2["ag_idx"],
                data_day2["RT"],
                data_day2["response"],
                data_day2['jokercondition'],
                data_day2["qdiff"],
                data_day2['repdiff']
            ]),  # Make sure this closing bracket matches with np.column_stack opening
            columns=["participant_id", "rt", "response", "jokercondition", "qdiff", "repdiff"]
        )     
    
    print("Check that the data for the days are returned correctly!")
    return ddmdata