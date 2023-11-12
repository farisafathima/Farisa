import pandas as pd
import numpy as np

def load_csv(filepath):
    return pd.read_csv(filepath)

def num_cols(dataframe):
    return len(dataframe.columns)

def num_rows(dataframe):
    return len(dataframe)

def fill_cols(dataframe):
    return dataframe.fillna(0) #its a method in python to fill nan values with some other num

