# %%

import pickle
from unittest import TestSuite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import OpperantBehaviorTools as obt
import json
import re

abspath = r"R:\Mike\BMAL_\test"  # path where raw med data is stored

# directory that contains raw data directories
raw_data_path = os.path.join(abspath, 'raw_data')
# gets list of mouse folders where raw data is stored.
aggreated_data_path = obt.create_new_dir(abspath, 'aggregated_data')
# creates new directory to save compiled data that is processed and analyzed
compiled_data_path = obt.create_new_dir(abspath, 'compiled_data')

parameters_path = obt.list_subdirs(r"R:\Mike\BMAL_\parameter_files")

# create list of all med files to be read
files = obt.list_subdirs(raw_data_path)
# %%
for f in files:  # read all med files with empty parameter file and save as feather files
    obt.read_medpc(f)

# %%


def read_params(path):
    """Function to load paremeters from json files -> to be used to rename columns in dataframe """

    with open(path) as f:
        params = json.load(f)
    return params


def get_names(f): return os.path.basename(f).split(
    '.')[0]  # function to get names of parameters


parameter = {get_names(f): read_params(f)
             for f in parameters_path}  # creat dict of names: parameters


def get_pr_data(df):
    """
    this function extracts the program features (ie. nosepokes) from the data,
    converts to its own column and concats to original dataframe.

    """

    data = pd.DataFrame(df['B'].dropna()).transpose()
    pr_cols = {
        0: "total_nosepokes",
        1: "correct_nosepokes",
        2: "incorrect_nosepokes",
        3: "total_rewards",
        4: "percent_correct",
        5: "percent_incorrect",
        6: "break_point",
        7: "port_status",
        8: "port_entries",
        9: "Counter for rewarded nose timestamps",
        10: "Counter for unrewarded nose on active side timestamps",
        11: "Counter for rewarded port entries",
        12: "Counter for unrewarded port entries",
        13: "Counter for Inactive NP timestamps"
    }

    pr_data = (
        data
        .rename(columns=pr_cols)
        .reset_index()
    )
    return pr_data


def rename_params(path):

    df = pd.read_csv(path)
    if 'FR' in df['MSN'][0]:
        df = df.rename(columns=parameter['AH_FR_parameters'])
    elif 'Progressive Ratio' in df['MSN'][0]:
        raw_df = df.rename(columns=parameter['AH_PR_parameters'])
        pr_data = get_pr_data(df)
        df = pd.concat([raw_df, pr_data], axis=1)
    df.to_csv(path)


read_files = obt.list_subdirs(raw_data_path)
read_files_csv = [f for f in read_files if f.endswith('csv')]

for f in read_files_csv:
    rename_params(f)

# %%


def find_param(params: list, regex: str,) -> str:
    for i in params:
        if re.match(regex, i):
            return i


def clean_fr(path):

    def get_protocol_features(protcocol: str, split_by='_| ') -> tuple:
        program_features = re.split(split_by, protcocol)
        side = find_param(program_features, regex='[Rr]ight|[Ll]eft')
        program = find_param(program_features, regex='FR[0-9]')
        return (side, program)

    df = pd.read_csv(path)

    cols = ['End Date', 'Subject', 'MSN',
            'Left_nosepokes', 'Left_Rewards',
            'Right_nosepokes', 'Right_Rewards',
            'Port_Entries']

    df = (
        df[cols]
        .rename(columns={'End Date': 'date', 'Subject': 'mouse_id', 'MSN': 'program'})
        .rename(columns=lambda c: c.lower())
        .assign(
            side=get_protocol_features(df['MSN'][0])[0],
            protocol=get_protocol_features(df['MSN'][0])[1],
            max_np=df.Right_nosepokes.where(
                df.Right_nosepokes > df.Left_nosepokes, other=df.Left_nosepokes),
            max_reward=df.Right_Rewards.where(
                df.Right_Rewards > df.Left_Rewards, other=df.Left_Rewards)

        )
        .dropna()

    )
    return df


def clean_pr(path):

    def get_side(protcocol: str, split_by='_| ', ) -> str:
        program_features = re.split(split_by, protcocol)
        side = find_param(program_features, regex='[Rr]ight|[Ll]eft')
        return side

    cols = ['End Date', 'Subject', 'MSN',
            'total_nosepokes', 'correct_nosepokes', 'incorrect_nosepokes',
            'total_rewards', 'percent_correct', 'percent_incorrect',
            'break_point', 'port_entries', ]
    df = pd.read_csv(path)
    df = (
        df[cols]
        .rename(columns={'End Date': 'date', 'Subject': 'mouse_id', 'MSN': 'program'})
        .rename(columns=lambda c: c.lower())
        .assign(protcol='PR',
                side=get_side(df['MSN'][0])
                )
        .dropna()
    )
    return df


# %%
"""TODO: 
* loop through read_csvs
* segreate by FR or Progressive Ratio
* apply clean FR or clean PR function and append to list
* concat lists as master dataframe
* save master CSV to aggregated data directory
"""


# %%

# %%
