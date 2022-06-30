# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import OpperantBehaviorTools as obt
import json
import re
"""
This section will create new directories to store extracted data, collect folders containing
MEDPC data, read data into pandas dataframe, and save as a feather file.


"""
# path where folder to data is stored
abspath = r'R:\Mike\BMAL_\test_data\cohort3'
# directory that contains raw data directories
raw_data_path = os.path.join(abspath, 'raw_data')
# gets list of mouse folders where raw data is stored.
raw_subject_folders = os.listdir(raw_data_path)
# creates new directory to save all extracted data seperate from raw data
extracted_data_path = obt.create_new_dir(abspath, 'extracted_data')
# creates new directory to save compiled data that is processed and analyzed
compiled_data_path = obt.create_new_dir(abspath, 'compiled_data')

parameters_path = os.path.join(
    abspath, 'parameters.json')  # path to parameters file
# %%
# loads json file with parameters
with open(parameters_path) as f:
    inputParameters = json.load(f)

# reads each file in subject folders, extracts med data and saves as csv in extracted_data_path
for i in range(len(raw_subject_folders)):
    folder = os.path.join(raw_data_path, raw_subject_folders[i])
    filepath = glob.glob(os.path.join(folder, "*"))

    for j in range(len(filepath)):
        obt.readMEDPC(filepath[j], inputParameters)

# %%
"""
This section will do analysis on extraxted data and re-save the file.

# TODO write analysis so columns are inserted into dataframe before timestamp columns

"""

# %%%
"""

This section will read the single files, exclude timestamps and concatenate them into a master datafram for data exploration and visualization.

"""
files = obt.list_subdirs(extracted_data_path)

df_to_concat = []
for f in files:
    df = pd.read_feather(f)
    df_truc = df.loc[0, :'Left_nosepokes']
    df_to_concat.append(df_truc)

comp = (
    pd.concat(df_to_concat, axis=1)
    .transpose()
    .reset_index(drop=True)
    .rename(columns=lambda c: c.replace(' ', '_'))
)
comp.to_feather(
    f'{compiled_data_path}/{os.path.basename(abspath)}_compiled_data.feather')
comp.to_csv(
    f'{compiled_data_path}/{os.path.basename(abspath)}_compiled_data.csv')


# test
# %%
df = pd.read_feather(
    r"R:\Mike\BMAL_\test_data\cohort3\compiled_data\cohort3_compiled_data.feather")


def tweak_opperant(df):
    return (
        df
        .assign(Subject=df.Subject.replace({'365.277': '356.277',
                                            '356.726': '356.276',
                                            '265.28': '365.28',
                                            '265.283': '365.283',
                                            }),
                protocol=df.MSN.str.replace('RK_[A-Z]_', '', regex=True)
                .str.split('_', expand=True)[0],
                sided=df.MSN.str.replace('RK_[A-Z]_', '', regex=True)
                                .str.split('_', expand=True)[1],
                max_np=df.Right_nosepokes.where(df.Right_nosepokes > df.Left_nosepokes,
                                                other=df.Left_nosepokes),
                max_reward=df.Right_Rewards.where(df.Right_Rewards > df.Left_Rewards,
                                                  other=df.Left_Rewards)

                )
    )


opperant = tweak_opperant(df)
opperant
# %%


# %%

def greater_than(col_1, col_2):
    if opperant.where(cond)


# %%

test = greater_than()

# %%
