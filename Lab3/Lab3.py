"""
Виконав: Васильєв Єгор
Lab_work_3, II група вимог
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_rows(df):
    """

    Parameters
    ----------
    df: the data to be normalized.

    Returns
    -------
    normalized_df: normalized (from 0 to 1) dataframe
    """
    normalized_df = df.copy()
    for index, row in normalized_df.iterrows():
        min_val = row.min()
        max_val = row.max()
        normalized_df.loc[index] = (row - min_val) / (max_val - min_val)
    return normalized_df


# data preparation
# noinspection PyTypeChecker
parsed_data = pd.read_excel("Lab3.xlsx", header=1, index_col=0, usecols="B:L", skiprows=0)
weights = parsed_data.iloc[:, -1] / 100
criteria = parsed_data.iloc[:, -2]
criteria = [True if x == 'min' else False for x in criteria]
data = parsed_data.drop(parsed_data.columns[-2:], axis='columns')

# data normalization
normalized_data = normalize_rows(data)
normalized_data.loc[criteria, :] = 1 - normalized_data.loc[criteria, :]  # checking the criteria type (min or max)
weighted_data = normalized_data.multiply(weights, axis='index')  # applying weigh coefficients

# result output
scores = weighted_data.sum()
print(f'All scores:\n', scores)
print(f'The best is {scores.idxmax()}')
print(f'The worst is {scores.idxmin()}')
