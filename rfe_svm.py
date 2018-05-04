import pandas as pd
import numpy as np
from scipy.stats import skewtest
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer

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
    tests.dropna(axis=0, inplace=True, subset=["IC50"])

    df = df.append(tests).reset_index(drop=True)
    y_data = df.pop("IC50")
    x_data = df.dropna(axis=1)  # Remove columns with missing values after combining
    return x_data, y_data


def load_compound_dataset():
    df = pd.read_csv("data/Series3_6.15.17_padel.csv")
    # df.IC50.fillna(-1, inplace=True)  # Mark potential compounds with -1
    df.dropna(axis=0, inplace=True, subset=["IC50"])
    y_data = df.pop("IC50")
    x_data = df.dropna(axis=1)
    return x_data, y_data


def preprocess_variables(x_data, remove_skewed=False):
    """ Preprocessing Variables """
    # Get dummy vars: filter to int type, convert to object, pass to get_dummies.
    assert not sum(x_data[x_data.columns[x_data.dtypes == 'int64']].isna().sum()), "Null values found in cat"
    cat_vars_df = pd.get_dummies(
        x_data[x_data.columns[x_data.dtypes == 'int64']].astype('O'))
    # Impute or remove? (for now remove any columns with nan)
    cont_vars_df = x_data[x_data.columns[x_data.dtypes == 'float64']].dropna(axis=1)
    # Remove skewed
    if remove_skewed:
        cont_vars_df = cont_vars_df.loc[:, cont_vars_df.apply(
            lambda x: skewtest(x)[1] > .05).values]
    # Combine datasets
    x_data = pd.concat([cat_vars_df, cont_vars_df], axis=1)
    return x_data


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
# Preprocess variables
full_x = preprocess_variables(full_x)
# Extract list of available columns
full_columns = full_x.columns
print("Loading in compound dataset....")
# Read in compound dataset
compound_x, compound_y = load_compound_dataset()
# Preprocess
compound_x = preprocess_variables(compound_x)
# Find intersecting features
avail_columns = compound_x.columns.intersection(full_columns)
# Select features on subset
x_data = compound_x.loc[:, avail_columns]
y_data = compound_y.copy()
# Create binary variable
y_class = np.squeeze([int(y_val <= 10) for y_val in y_data])

print("Adding non-linear features to compound dataset....")
# Add all transformations on compound data
for feature in x_data.columns[x_data.dtypes == 'float64']:
    x_data = add_transformations(x_data, feature)
# Drop any new columns with NaN due to improper transformation
x_data.replace([np.inf, -np.inf], np.nan, inplace=True)
x_data.dropna(axis=1, inplace=True)
assert not sum(x_data.isna().sum()), "Unexpected nulls found"

# Perform classification with SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

model = SVC(kernel="linear", class_weight={0:1-np.mean(y_class), 1: np.mean(y_class)}, random_state=0, probability=True)
model.fit(x_data, y_class)

best_features = [f for i, f in sorted(zip(model.coef_[0], x_data.columns), reverse=True) if i != 0]

accuracy_score(y_class, model.predict(x_data))
pred = cross_val_predict(model, x_data, y_class, cv=sum(y_class), method="predict_proba")

# TODO Develop a cost sensitive metric
# TODO Create augmented validation set
# Penalize the miss more-so the higher the actual potency


# Backward step-wise algorithm
# https://pdfs.semanticscholar.org/00a4/7b1031b0fdb0c2b7035a51917feeb189aa26.pdf
# Section 4 ^^
# Compromize between speed and feature quality, remove 10% of current features every iteration
# Validation set: consider creating an augmented dataset that is derived from the potent compounds


# TODO Geometric mean for model comparison

# http://cs229.stanford.edu/notes/cs229-notes3.pdf
