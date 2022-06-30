
import os
import numpy as np
import re
import glob
import json
import pandas as pd
from collections import OrderedDict
from pathlib import Path
import feather

""" 
functions for directory management

"""


def create_new_dir(file_path, new_dir_ext):
    """ creates new directory from file path and extension"""
    new_directory = os.path.join(file_path, new_dir_ext)
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)
        print('directory created')
    else:
        print('directroy already exists')
    return new_directory


def list_subdirs(directory):
    """ given directory, returns list of all sub dirs as full path"""
    return [os.path.join(directory, file) for file in os.listdir(directory)]


"""

functions for reading and aggregating MEDPC data

"""


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def stripLine(line, index, inputParameters):
    arr = []
    res = line[index].strip(":\n").split(": ")

    key = None
    value = None

    if len(res) > 1:
        key = res[0]
        value = res[1]
        if res[0] in inputParameters.keys():
            key = inputParameters[res[0]]
        index += 1
    else:

        if res[0] in inputParameters.keys():
            key = inputParameters[res[0]]
        else:
            key = res[0]

        index += 1

        value = []
        strip = line[index].strip().split(" ")

        while len(strip) > 1 and index < len(line)-1:
            for i in strip[1:]:
                if len(i) > 0:
                    value.append(i)
            index += 1

            strip = line[index].strip().split(" ")

    if key:
        if type(value) is list:
            return key, np.asarray(value, dtype=np.float32), index
        else:
            return key, value, index
    else:
        return 0, 0, index


def findStartLocation(data):
    location = []
    for i in range(len(data)):
        if i == len(data)-1:
            location.append(i)
            break
        if "Start Date" in data[i]:
            location.append(i)

    print(location)
    return location


def readMEDPC(filepath, inputParameters, parent_level=2):
    path = Path(filepath)
    extracted_path = path.parents[parent_level]
    path_to_save = os.path.join(extracted_path, 'extracted_data')

    res = OrderedDict()

    print(filepath)
    with open(filepath, "r") as in_file:
        line = in_file.readlines()

    location = findStartLocation(line)
    for i in range(1, len(location)):
        j = location[i-1]
        while j >= location[i-1] and j < location[i]:
            key, value, index = stripLine(line, j, inputParameters)
            res[key] = value
            j = index

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in res.items()]))
        df.to_feather(os.path.join(
            path_to_save, res["Subject"]+"_"+res["Start Date"].replace("/", "_")+".feather"))
