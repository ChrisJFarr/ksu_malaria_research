import pandas as pd
import numpy as np
from scipy.stats import skewtest
import os

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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import pickle
from par_support import par_backward_stepwise
from time import time
"""
Steps taken:
1. Preprocess features with non-linear transformations
2. Tune SMOTE/downsampling and model params using SVM with all features
3. Create list of splits, already preprocessed with SMOTE/downsampling
4. Create list of splits, w/o SMOTE/downsampling
5. W/ SMOTE splits, start with all features, score the removal of every feature with ROC, 
iteratively remove lowest 10%
"""

# Forget smote, just use backward stepwise selection
# Score using log_loss and roc (compare results)
scaler = StandardScaler()
# Scale the train data
x_train = scaler.fit_transform(x_data)
x_train = pd.DataFrame(data=x_train, columns=x_data.columns, index=x_data.index)

# # Set params for tuning
model = SVC(random_state=0, probability=True)
params = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
          "C": np.arange(0.05, 1.05, .05),
          "class_weight": [None, "balanced"]}

grid = GridSearchCV(model, param_grid=params, scoring=make_scorer(roc_auc_score),
                    cv=sum(y_class), n_jobs=7)
grid.fit(x_train, y_class)
print(grid.best_params_)

# Baseline should be decent

scaler = StandardScaler()
# Scale the train data
x_train = scaler.fit_transform(x_data)
x_train = pd.DataFrame(data=x_train, columns=x_data.columns, index=x_data.index)

# TODO try with adaboost too
# Refine params to keep somewhat consistent during process
model = SVC(random_state=0, class_weight="balanced", kernel="sigmoid", probability=True, C=0.95)

benchmark = np.mean(cross_val_score(model, x_train, y_class,
                                    scoring=make_scorer(roc_auc_score),
                                    cv=sum(y_class), n_jobs=7))

# def par_function(features_in, x_data, y_data, model):
#     feature_dict = dict()
#     for out_feat in features_in:
#         iter_features = [feat for feat in features_in if feat != out_feat]
#         # Score the removal of feature
#         score = np.mean(cross_val_score(model, x_data[iter_features], y_data,
#                                         scoring=make_scorer(log_loss, needs_proba=True),
#                                         cv=5, n_jobs=5))
#         # Set result in feature dict
#         feature_dict[out_feat] = score
#     return feature_dict


# return_dict = par_function(features_in[0:5], x_train, y_class, model)

# The worst features have the highest scores, select top 10 percent and remove from full list
# sorted_features = sorted(return_dict.keys(), key=lambda x: return_dict[x], reverse=True)
# sorted_features

# Set benchmark for each round, only remove those that lowered loss, and were

# Create splits for parallel jobs

# Start with random selection of columns?
starting_benchmark = benchmark
start_time = time()
features_in = list(x_train.columns)

while True:  # While features to remove, add break
    higher_is_better = True
    start = time()
    results_par = Parallel(n_jobs=7)(
        delayed(par_backward_stepwise)(features, x_train, y_class, model) for features in np.array_split(features_in, 7))
    norm_summary = [item for sublist in results_par for item in sublist]
    stop = time()
    print((stop - start))

    # TODO update benchmark
    return_dict = dict()
    for d in results_par:
        # assert isinstance(d, dict)
        for k, v in d.items():
            return_dict[k] = v

    # TODO If list len is 0 then stop, no more features to remove
    # Intuition: greater than benchmark means removal increased the loss function
    if higher_is_better:
        potential_removals = {feat: return_dict[feat] for feat, roc in return_dict.items() if roc >= benchmark}
    else:
        potential_removals = {feat: return_dict[feat] for feat, roc in return_dict.items() if roc <= benchmark}
    if len(potential_removals) == 0:
        print("nothing to remove")
        break

    # Remove features with the best scores (top 10%) intuition: removing them led to improved scores
    filter_to = max(len(return_dict) - len(potential_removals), int(len(return_dict) * .90))
    features_in = sorted(return_dict.keys(), key=lambda x: return_dict[x], reverse=higher_is_better)[:filter_to]

    # Set benchmark
    benchmark = np.mean(cross_val_score(model, x_train[features_in], y_class,
                                        scoring=make_scorer(roc_auc_score),
                                        cv=sum(y_class), n_jobs=7))

selected_features = features_in.copy()


pickle.dump(selected_features, open("selected_features.pkl", "wb"))
finish_time = time()
print("total time %s" % str(finish_time - start_time))
print("final score: %s" % str(benchmark))
len(selected_features)

# Test outcome

# Analyze CV prediction performance
predict = cross_val_predict(
    model, x_data[selected_features], y_class, cv=sum(y_class), method="predict")

print(confusion_matrix(y_class, predict, labels=[1, 0]))
print(np.array([["TP", "FN"], ["FP", "TN"]]))

# [[ 8  3]
#  [24 12]]
