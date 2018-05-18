# Py sheet for performing feature selection with a high performance model with
# feature transformations that intersect with available data on the Selleck dataset.
# The key to this model will be ensuring that the performance doesn't drop due to the
# limited features available between the two datasets.

import pandas as pd
import numpy as np
from scipy.stats import skewtest
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import roc_auc_score, make_scorer, fbeta_score
from joblib import Parallel, delayed
from custom_functions import *

avail_transformations = ["log", "log2", "log10", "cubert",
                         "sqrt", "exp", "exp2", "cube", "sq"]

print("Loading in Selleck dataset....")
# Load in Selleck dataset
selleck = pd.read_csv("data/Imputed_Selleck_filtered_padel_corrected.csv")
selleck_names = selleck.Name
selleck_x = selleck.drop("Name", axis=1)
assert not sum(selleck_x.isna().sum()), "Unexpected nulls found"
# Remove 0 variance features
selleck_x = selleck_x.loc[:, selleck_x.std() != 0].copy()
# Extract list of available columns
selleck_columns = selleck_x.columns
print("Loading in compound dataset....")
# Read in compound dataset
compounds = pd.read_csv("data/Series3_6.15.17_padel.csv")
compounds.dropna(axis=0, inplace=True, subset=["IC50"])
y_data = compounds.pop("IC50")
x_data = compounds.dropna(axis=1)
# Find intersecting features
avail_columns = x_data.columns.intersection(selleck_columns)
print("%s columns are available between datasets" % len(avail_columns))
# Select features on subset
x_data = x_data.loc[:, avail_columns]
print("Adding non-linear features to compound dataset....")
# Add all transformations on compound data
for feature in x_data.columns[x_data.dtypes == 'float64']:
    x_data = add_transformations(x_data, feature)
    selleck_x = add_transformations(selleck_x, feature)

# Drop any new columns with NaN due to improper transformation
x_data.replace([np.inf, -np.inf], np.nan, inplace=True)
x_data.dropna(axis=1, inplace=True)
selleck_x.replace([np.inf, -np.inf], np.nan, inplace=True)
selleck_x.dropna(axis=1, inplace=True)

# Align columns again after tranformations
avail_columns = x_data.columns.intersection(selleck_x.columns)
print("%s total columns are available" % len(avail_columns))
assert not sum(x_data.isna().sum()), "Unexpected nulls found"
x_data = x_data.loc[:, avail_columns].copy()
selleck_x = selleck_x.loc[:, avail_columns].copy()
assert set(selleck_x.columns) == set(x_data.columns), "Mismatched columns"

# Create binary variables at different break points
print("Building binary target variables....")
y_class_10 = np.squeeze([int(y_val <= 10) for y_val in y_data])
y_class_25 = np.squeeze([int(y_val <= 25) for y_val in y_data])
y_class_40 = np.squeeze([int(y_val <= 40) for y_val in y_data])

# Generating predictions for Selleck
df = selleck_names.to_frame()

# Tune AdaBoost Classifier on each class set
print("Tuning AdaBoost on compound dataset....")


def tune_classifier(x_data, y_class):
    model = AdaBoostClassifier(random_state=0)
    params = {"n_estimators": [5, 7, 10],
              "learning_rate": [0.05, 0.075]}
    grid = GridSearchCV(estimator=model, param_grid=params,
                        cv=sum(y_class), n_jobs=7, scoring=make_scorer(roc_auc_score))
    grid.fit(x_data, y_class)
    best_model = grid.best_estimator_
    return best_model


# Tune/validate/train/test on lte 10
model = tune_classifier(x_data, y_class_10)
predict = cross_val_predict(model, x_data, y_class_10, cv=sum(y_class_10), method="predict")
print(confusion_matrix(y_class_10, predict, labels=[1, 0]))
print(np.array([["TP", "FN"], ["FP", "TN"]]))
print("ROC score: %s" % roc_auc_score(y_class_10, predict))
model.fit(x_data, y_class_10)
predict = model.predict(selleck_x)
df["adaboost_0.748_cv_roc_lte_10"] = predict

# Tune/validate/train/test on lte 25
model = tune_classifier(x_data, y_class_25)
predict = cross_val_predict(model, x_data, y_class_25, cv=sum(y_class_25), method="predict")
print(confusion_matrix(y_class_25, predict, labels=[1, 0]))
print(np.array([["TP", "FN"], ["FP", "TN"]]))
print("ROC score: %s" % roc_auc_score(y_class_25, predict))
model.fit(x_data, y_class_25)
predict = model.predict(selleck_x)
df["adaboost_0.746_cv_roc_lte_25"] = predict

# Tune/validate/train/test on lte 40
model = tune_classifier(x_data, y_class_40)
predict = cross_val_predict(model, x_data, y_class_40, cv=sum(y_class_40), method="predict")
print(confusion_matrix(y_class_40, predict, labels=[1, 0]))
print(np.array([["TP", "FN"], ["FP", "TN"]]))
print("ROC score: %s" % roc_auc_score(y_class_40, predict))
model.fit(x_data, y_class_40)
predict = model.predict(selleck_x)
df["adaboost_0.67_cv_roc_lte_40"] = predict

# Write to CSV
df.to_csv("selleck_predictions_chris.csv")
df.head()
