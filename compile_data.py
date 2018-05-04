import pandas as pd
import numpy as np
from scipy.stats import skewtest
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
import pickle

""" Load and combine all datasets """


def load_full_dataset():
    file_list = os.listdir("data")
    file_list = [f for f in file_list if "decoys" in f and f.endswith(".csv")]
    # df = pd.DataFrame()  # Initialize for IDE warnings
    df = None
    for f in file_list:
        print("Adding " + f + "....")
        new = pd.read_csv("data/" + f)
        if df is None:
            df = new.copy()
        else:
            df = df.append(new)
    df["IC50"] = 250  # Try 100 and 250
    tests = pd.read_csv("data/Series3_6.15.17_padel.csv")
    df = df.append(tests).reset_index(drop=True)
    y_data = df.pop("IC50")
    x_data = df.copy()
    return x_data, y_data


def load_compound_dataset():
    df = pd.read_csv("data/Series3_6.15.17_padel.csv")
    y_data = df.pop("IC50")
    x_data = df.copy()
    return x_data, y_data


def add_transformations(data, feat):
    """
    Input single featuer dataframe, return feature with added transformations
    :param data:
    :param feat:
    :return:
    """
    assert isinstance(data, pd.DataFrame), "Input must be a dataframe"
    feature_df = data.loc[:, feat].copy()
    if feature_df.min() > 0:  # Avoid 0 or negative
        data.loc[:, feat + "_log"] = feature_df.apply(np.log)  # log
        data.loc[:, feat + "_log2"] = feature_df.apply(np.log2)  # log2
        data.loc[:, feat + "_log10"] = feature_df.apply(np.log10)  # log10
    data.loc[:, feat + "_cubert"] = feature_df.apply(
        lambda x: np.power(x, 1 / 3))  # cube root
    data.loc[:, feat + "_sqrt"] = feature_df.apply(np.sqrt)  # square root
    # Avoid extremely large values, keep around 1M max
    if feature_df.max() < 13:
        data.loc[:, feat + "_exp"] = feature_df.apply(np.exp)  # exp
    if feature_df.max() < 20:
        data.loc[:, feat + "_exp2"] = feature_df.apply(np.exp2)  # exp2
    if feature_df.max() < 100:
        data.loc[:, feat + "_cube"] = feature_df.apply(
            lambda x: np.power(x, 3))  # cube
    if feature_df.max() < 1000:
        data.loc[:, feat + "_sq"] = feature_df.apply(np.square)  # square
    return data


def add_single_transformation(data, base_feat, transformation):
    if transformation == "log":
        data.loc[:, base_feat + "_log"] = data.loc[:, base_feat].apply(np.log)
    elif transformation == "log2":
        data.loc[:, base_feat + "_log2"] = data.loc[:, base_feat].apply(np.log2)
    elif transformation == "log10":
        data.loc[:, base_feat + "_log10"] = data.loc[:, base_feat].apply(np.log10)
    elif transformation == "cubert":
        data.loc[:, base_feat + "_cubert"] = data.loc[:, base_feat].apply(
            lambda x: np.power(x, 1 / 3))
    elif transformation == "sqrt":
        data.loc[:, base_feat + "_sqrt"] = data.loc[:, base_feat].apply(np.sqrt)
    elif transformation == "exp":
        data.loc[:, base_feat + "_exp"] = data.loc[:, base_feat].apply(np.exp)
    elif transformation == "exp2":
        data.loc[:, base_feat + "_exp2"] = data.loc[:, base_feat].apply(np.exp2)
    elif transformation == "cube":
        data.loc[:, base_feat + "_cube"] = data.loc[:, base_feat].apply(
            lambda x: np.power(x, 3))  # cube
    elif transformation == "sq":
        data.loc[:, base_feat + "_sq"] = data.loc[:, base_feat].apply(np.square)
    else:
        print("No transformation performed, check `transformation` input: %s" % transformation)
    return data


avail_transformations = ["log", "log2", "log10", "cubert", "sqrt", "exp", "exp2", "cube", "sq"]

# Problem: features selected in lasso aren't all in any single decoy set
# Solution: Perform lasso selection with only features available across all decoys
# Load in full dataset
full_x, full_y = load_full_dataset()
# Extract list of available columns
full_columns = full_x.columns
print("Loading in compound dataset....")
# Read in compound dataset
compound_x, compound_y = load_compound_dataset()
# Find intersecting features
avail_columns = compound_x.columns.intersection(full_columns)
# Select features on subset
x_data = compound_x.loc[:, avail_columns].drop("Name", axis=1)
y_data = compound_y.copy()

# Pickle datasets
pickle.dump(x_data, open("x_data.pkl", "wb"))
pickle.dump(y_data, open("y_data.pkl", "wb"))

# TODO Load datasets
x_data = pickle.load(open("x_data.pkl", "rb"))
y_data = pickle.load(open("y_data.pkl", "rb"))
