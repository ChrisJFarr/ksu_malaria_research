"""
The most active compound is called OSM-S-106. However, we do not understand how OSM-S-106 works to kill the 
malaria parasite. We wish to identify the target of OSM-S-106 in the parasite. <b>Knowing the target will help us create
more potent versions of OSM-S-106.</b>

We are in the process of performing experiments with OSM and KU to identify the OSM-S-106 target. Experiments are 
slow and very expensive. We would also like to apply machine learning methods to predict potential targets. To do 
that, we have calculated molecular descriptors, which describe chemical features of the drug candidate molecules. 

We wish to find descriptors that would help predict potency (described by the "IC50").

Questions we want to research: 
* Which descriptors best predict potency? Our data set is very small. Finding an 
effective drug is like finding a needle in a haystack. This is a common problem with scientific data sets. 
* Can we augment the data set with predicted negative data (molecules expected to be inactive) to improve our machine
learning models?
* Are there certain characteristics of negative data sets that are the most useful for training? 
* Given the limited size of the data set and the high cost of experiments, can we use ML to identify the missing data 
that would be best for model training? In this way, ML would be recommending future experiments. Apply the ML model to 
set of well-characterized drugs. 
* Which cluster most closely with OSM-S-106? Would this provide clues as to the mechanism of OSM-S-106? 
How well do more advanced ML models perform over simple methods like multiple linear regression, SVM, and random forest? 



What is the activate compound (OSM-S-106) targeting within the malaria parasite?
Leverage experiment results and molecular descriptors of effective drug.
What dimensions are accurate predictors of "potency".




Is this feature a predictor of potency?
Scaling the feature and creating a new target that is an average of the potency times the presence of the characteristic

# How well can we predict the performance of the target without using it to train?

"""
import pandas as pd
from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import skewtest
import os
import pandas as pd
import gc

# Read in dataset
df = pd.read_csv("data/Series3_6.15.17_padel.csv")

""" Preprocessing Variables """
# Get dummy vars: filter to int type, convert to object, pass to get_dummies.
cat_vars_df = pd.get_dummies(
    df[df.columns[df.iloc[:, :].dtypes == 'int64']].astype('O'))

# Impute or remove? (for now remove any columns with nan)
cont_vars_df = df[df.columns[df.iloc[:, :].dtypes == 'float64']].dropna(axis=1)
# Note: Target variable is dropped from both dataframes and only in df


def add_transformations(data, feat):
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


# Add above transformations for all continuous variables
for feature in cont_vars_df.columns:
    cont_vars_df = add_transformations(cont_vars_df, feature)
# Drop any new columns with NaN due to improper transformation
cont_vars_df.replace([np.inf, -np.inf], np.nan, inplace=True)
cont_vars_df.dropna(axis=1, inplace=True)

# Assume skewed if we can reject the null hypothesis with 95% certainty
# Remove any skewed features after adding transformations
cont_vars_df = cont_vars_df.loc[:, cont_vars_df.apply(
    lambda x: skewtest(x)[1] > .05).values]

# Combine datasets
x_data = pd.concat([cat_vars_df, cont_vars_df], axis=1)
y_data = df.IC50

# Remove missing IC50
x_no_ic50 = x_data.loc[df.IC50.isnull(), :]
x_no_ic50_names = df.loc[df.IC50.isnull(), "Name"]

# Perform forward stepwise feature selection
x_data = x_data.loc[~df.IC50.isnull(), :].reset_index(drop=True)
y_data = y_data.loc[~df.IC50.isnull()].reset_index(drop=True)

# Select features that are the best at predicting OSM-S-106


# Loop through each column, train on all but 0, test on 0
cont_vars = x_data.columns[x_data.iloc[:, :].dtypes == 'float64']

# Train model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error


best_feat = None
best_score = np.inf
# TODO Start here! Working through step-wise feature selection
# TODO continue loop by adding to list of features and tracking what haven't been added yet
for feat in cont_vars[0:100]:
    feat = best_feat
    x_train, y_train = x_data.loc[1:, feat], y_data.loc[1:]
    x_test, y_test = x_data.loc[0, feat], y_data.loc[0]

    # Scale
    x_scaler, y_scaler = StandardScaler(), MinMaxScaler(feature_range=(.01, .99))
    x_train = x_scaler.fit_transform(x_train.values.reshape(-1, 1))
    x_test = x_scaler.transform(x_test)
    y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test = y_scaler.transform(y_test)

    # Model
    model = MLPRegressor(hidden_layer_sizes=(1,), activation="logistic",
                         max_iter=1000, learning_rate="adaptive")
    model.fit(x_train, y_train.ravel())
    pred = model.predict(x_test)
    # train_pred = model.predict(x_train)

    mse = mean_squared_error(y_scaler.inverse_transform(y_test.reshape(-1, 1)),
                             y_scaler.inverse_transform(pred.reshape(-1, 1)))
    rmse = np.sqrt(mse)

    if rmse < best_score:
        best_feat = feat
        best_score = rmse







""" Load and combine all datasets """
file_list = os.listdir("data")
file_list = [f for f in file_list if "decoys" in f and f.endswith(".csv")]

df = pd.read_csv("data/" + file_list[0])
for f in file_list[1:]:
    df = df.append(pd.read_csv("data/" + f))

# Combine with Series3_6.15.17_padel.csv
tests = pd.read_csv("data/Series3_6.15.17_padel.csv")
tests.IC50.fillna(-1, inplace=True)  # Mark potential compounds with -1

df = df.append(tests).reset_index(drop=True)
y_data = df.pop("IC50")
x_data = df.dropna(axis=1)
del df
gc.collect()

# Primarily floats have complete features across datasets
print("%s continuous features" % sum(x_data.dtypes == "float64"))
print("%s categorical features" % sum(x_data.dtypes == "int64"))

""" Preprocessing """

""" Process Continuous Variable """


def add_transformations(data, feat):
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

# Add above transformations for all continuous variables
for feature in x_data.columns:
    if x_data.loc[:, feature].dtype == "float64":
        x_data = add_transformations(x_data, feature)

# Parallel Transformations




# Drop any new columns with NaN due to improper transformation
cont_vars_df.replace([np.inf, -np.inf], np.nan, inplace=True)
cont_vars_df.dropna(axis=1, inplace=True)

# Assume skewed if we can reject the null hypothesis with 95% certainty
# Remove any skewed features after adding transformations
cont_vars_df = cont_vars_df.loc[:, cont_vars_df.apply(
    lambda x: skewtest(x)[1] > .05).values]

# Combine datasets
vars_df = pd.concat([cat_vars_df, cont_vars_df], axis=1)
