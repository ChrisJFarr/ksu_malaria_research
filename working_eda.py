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

* Can we augment the data set with predicted negative data (molecules expected to be inactive) to improve our machine
learning models? TODO Next step here

Approach:
* Full data performance compared to compounds only; no dimension reduction
* Compare model performance by testing the OSM-S-106 prediction accuracy
* Full: Neural network, XGBoost
* Small: Lasso regression, nnet, LDA, Tree boosting

"""
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

# Smote
from custom_pipe_helper import SMOTER

import auto

smote = SMOTE()

check = smote.fit(x_data, y_class)
smote.fit_sample()
check = smote.sample(x_data, y_class)

check[0].shape
check[1]

# Create folds
# For each fold
# SMOTE the train data
# Train model
# Evaluate model

from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import itertools as it
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.under_sampling import RandomUnderSampler

# Custom grid search for best params with SMOTE on train
params = {"a_SMOTE_k_neighbors": [2, 3, 4],
          "b_SMOTE_m_neighbors": [9, 10, 11],
          "c_model_n_estimators": [35, 40, 45],
          "d_model_learning_rate": [0.05, 0.075],
          "smote": [True, False],
          "downsample": [False, 1, 2, 3, 4]}
all_names = sorted(params)  # TODO remove sorted but ensure correct order
combinations = it.product(*(params[Name] for Name in all_names))

# Initialize params
best_score = -np.inf
best_params = None

for param_values in combinations:
    param_dict = dict(zip(all_names, param_values))

    model = AdaBoostClassifier(random_state=0, learning_rate=param_dict.get("d_model_learning_rate"),
                               n_estimators=param_dict.get("c_model_n_estimators"))
    smote = SMOTE(random_state=0, k_neighbors=param_dict.get("a_SMOTE_k_neighbors"),
                  m_neighbors=param_dict.get("b_SMOTE_m_neighbors"))
    kfold = StratifiedKFold(n_splits=sum(y_class), shuffle=True, random_state=0)
    prediction_df = pd.DataFrame(columns=["prediction"])
    scores = []
    for train, test in kfold.split(x_data, y_class):
        # Split into train/test
        x_train, y_train = x_data.iloc[train], y_class[train]
        x_test, y_test = x_data.iloc[test], y_class[test]
        assert sum(y_test) == 1, "Ensure only one positive class is in test per iteration"
        # Perform SMOTE transformation on train
        if param_dict.get("smote"):
            x_train, y_train = smote.fit_sample(x_train, y_train)
        # Downsample randomly
        if param_dict.get("downsample"):
            negative_n = len(y_train) - sum(y_train)
            positive_n = sum(y_train)
            balance = min(1, param_dict.get("downsample") * (positive_n / negative_n))
            negative_n *= balance
            negative_n = int(negative_n)
            down_sampler = RandomUnderSampler(ratio={0: negative_n, 1: positive_n}, random_state=0)
            x_train, y_train = down_sampler.fit_sample(x_train, y_train)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        # sample_weight = compute_sample_weight("balanced", y_train)
        scores.append(roc_auc_score(y_test, prediction))
    # Calculate scores for parameters
    params_score = np.mean(scores)
    # Test new score against best
    if params_score > best_score:
        # Store the best
        best_parms = param_dict
        best_score = params_score

check = {"a": 0, "b": 4}

check2 = check

check2.get("a")
del check
# TODO Save for implementation
prediction = model.predict(x_test)
prediction_df = prediction_df.append(pd.DataFrame({"prediction": prediction}, index=test))
prediction_df.sort_index(inplace=True)
prediction_df = prediction_df.astype(int)


print(confusion_matrix(y_class, prediction_df, labels=[1, 0]))
print(np.array([["TP", "FN"], ["FP", "TN"]]))

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

model = SVC(kernel="linear", class_weight={0: 1 - np.mean(y_class), 1: np.mean(y_class)}, random_state=0,
            probability=True)
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


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, make_scorer
# How well does AdaBoost predict potency?
print("Tuning AdaBoost on compound dataset....")
model = AdaBoostClassifier(random_state=0)
params = {"n_estimators": [35, 40, 45],
          "learning_rate": [0.05, 0.075]}
grid = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=3, scoring=make_scorer(roc_auc_score))

grid.fit(x_data, y_class)
print(grid.best_params_)
best_model = grid.best_estimator_