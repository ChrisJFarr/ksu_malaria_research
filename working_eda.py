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
from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import skewtest
import os
import pandas as pd
import gc
import par_support
from importlib import reload

reload(par_support)


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
    df["IC50"] = 100  # Try 250 too
    # df.dropna(axis=0, inplace=True)  # Remove rows with missing values
    # Combine with Series3_6.15.17_padel.csv
    tests = pd.read_csv("data/Series3_6.15.17_padel.csv")
    tests.IC50.fillna(-1, inplace=True)  # Mark untested compounds with -1

    df = df.append(tests).reset_index(drop=True)
    y_data = df.pop("IC50")
    x_data = df.dropna(axis=1)  # Remove columns with missing values after combining
    return x_data, y_data


def load_compound_dataset():
    df = pd.read_csv("data/Series3_6.15.17_padel.csv")
    df.IC50.fillna(-1, inplace=True)  # Mark potential compounds with -1
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


# Problem: features selected in lasso aren't all in any single decoy set
# Solution: Perform lasso selection with only features available across all decoys
# Load in full dataset
full_x, full_y = load_full_dataset()
# Preprocess variables
full_x = preprocess_variables(full_x)
# Extract list of available columns
full_columns = full_x.columns
# Read in compound dataset
compound_x, compound_y = load_compound_dataset()
# Preprocess
compound_x = preprocess_variables(compound_x)
# Find intersecting features
avail_columns = compound_x.columns.intersection(full_columns)
# Select features on subset
x_data = compound_x.loc[:, avail_columns]
y_data = compound_y.copy()

assert not sum(x_data.isna().sum()), "Unexpected nulls found"
# Add all transformations on compound data
for feature in x_data.columns[x_data.dtypes == 'float64']:
    x_data = add_transformations(x_data, feature)
# Drop any new columns with NaN due to improper transformation
x_data.replace([np.inf, -np.inf], np.nan, inplace=True)
x_data.dropna(axis=1, inplace=True)

# Remove likely skewed features
non_skewed = x_data.apply(lambda x: skewtest(x)[1] > .05).values
cat_vars = (x_data.dtypes == 'int64').values
features = [non_skew | int_type for non_skew, int_type in zip(non_skewed, cat_vars)]
x_data = x_data.loc[:, features].copy()

# Remove untested compounds
y_data = y_data[y_data.values != -1].copy()
x_data = x_data.loc[y_data.values != -1].copy()

# Perform lasso selection on compounds
# Normalize variables
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_train = x_scaler.fit_transform(x_data)
x_columns = list(x_data.columns)
y_train = y_scaler.fit_transform(y_data.values.reshape(-1, 1))

""" Perform lasso selection """
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.3, max_iter=100000, tol=1e-5)
model.fit(x_train, y_train)
# Extract coefficients
positive_coeffs = len([c for c in model.coef_ if c > 0])
neg_coeffs = len([c for c in model.coef_ if c < 0])
# All non-zero are selected as predictive indicators
pred_indicators = [f for f, c in zip(x_columns, model.coef_) if c != 0]
pred_indicators = [col in pred_indicators for col in x_data.columns]
print(sum(pred_indicators))


def create_map(x):
    if x < 1:
        return 'black'
    elif x < 10:
        return 'blue'
    elif x < 15:
        return 'deepskyblue'
    else:
        return 'lightgreen'


color_map = list([create_map(i) for i in y_data])

""" Build PCA """
from sklearn.decomposition import KernelPCA
# Build Polynomial Pricipal Components, include all dimensions
pca = KernelPCA(n_components=2, kernel="linear", random_state=0, n_jobs=3)
pca_out = pca.fit_transform(x_train[:, pred_indicators])

# Sort pca so potent show on top of non-potent (easier visibility)
sorted_pca = [k for (j, k) in sorted(zip(y_train, pca_out),
                                     key=lambda pair: pair[0], reverse=True)]
color_map = [k for (j, k) in sorted(zip(y_train, color_map),
                                    key=lambda pair: pair[0], reverse=True)]
sorted_pca = np.array(sorted_pca)
x_val = sorted_pca[:, 0]
y_val = sorted_pca[:, 1]
plt.scatter(x=x_val, y=y_val, c=color_map, alpha=1)
plt.xlim(min(x_val) * 1.1, max(x_val) * 1.1)
plt.ylim(min(y_val) * 1.1, max(y_val) * 1.1)
plt.title("Polynomial PCA")
plt.show()


""" Extract coefficients from PCA components """
# What separates the black dot from the green the most?
pca = KernelPCA(n_components=None, kernel="linear", random_state=0, n_jobs=3)
<<<<<<< HEAD
pca_train = pca.fit_transform(x_data.loc[:, pred_indicators])
# Create classification y var
y_class = np.squeeze([int(y_val < 10) for y_val in y_data])

# Logistic regression
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0, class_weight={0: 1-np.mean(y_class), 1: np.mean(y_class)})
params = {"penalty": ["l1", "l2"], "C":[0.1, 0.3, 0.5, .9, 1.]}
grid = GridSearchCV(model, params)
grid.fit(pca_train, y_class)
best_model = grid.best_estimator_

best_model.predict
=======
pca_out = pca.fit_transform(compound_x.loc[:, pred_indicators])

""" Extract feature importance from PCA components """
# Try at least two approaches to compare importance outcomes
# XGBOOST and another
from sklearn.linear_model import LinearRegression
model = Lasso(alpha=0.01, max_iter=100000, tol=1e-5)
model.fit(pca_out, y_train)

np.max(model.coef_)

>>>>>>> 0276f2ff1a6b6225aa14636b04a1c13b92d2a480
""" Plot top 2 coefficients (most important features """
model.fit(pca_train, y_class)
train_pred = best_model.predict_proba(pca_train)
train_pred = best_model.predict(pca_train)
# CV Score is inaccurate at this point.
cv_pred = cross_val_predict(best_model, pca_train, y_class, cv=5)
# Analysis:
"""
With lasso feature selection, and starting with only the columns common across the decoys
and compound dataset, finding a stable set of features was challenging. 
Higher lasso penatly stabalized the PCA, but too high resulted in all zero coefficients.
The prediction model using lasso selected features was unable to differentiate between classes of 
potent (IC50<10) and non-potent.

<<<<<<< HEAD
"""

coeffs = model.coef_
# Sort from smallest to largest
ranked_coeffs = [i for k, i in sorted(zip(np.squeeze(coeffs), range(len(coeffs[0]))))]
len(coeffs[0])
""" Build PCA """
from sklearn.decomposition import KernelPCA
# Build Polynomial Pricipal Components, include all dimensions
pca = KernelPCA(n_components=2, kernel="linear", random_state=0, n_jobs=3)
pca_out = pca.fit_transform(x.loc[:, pred_indicators])

# Sort pca so potent show on top of non-potent (easier visibility)
color_map = list([create_map(i) for i in y])

sorted_pca = [k for (j, k) in sorted(zip(y.values, pca_out),
                                     key=lambda pair: pair[0], reverse=True)]
color_map = [k for (j, k) in sorted(zip(y.values, color_map),
                                    key=lambda pair: pair[0], reverse=True)]
sorted_pca = np.array(sorted_pca)
x_val = sorted_pca[:, 0]
y_val = sorted_pca[:, 1]
plt.scatter(x=x_val, y=y_val, c=color_map, alpha=1)
plt.xlim(min(x_val) * 1.1, max(x_val) * 1.1)
plt.ylim(min(y_val) * 1.1, max(y_val) * 1.1)
plt.title("Polynomial PCA")
plt.show()
=======
print("Number of predictors: %s" % len(pred_indicators))
# Assume skewed if we can reject the null hypothesis with 95% certainty
# Remove any skewed features after adding transformations

# PCA

# Model
# Can we make an accurate (define accurate?) prediction on OSM-S-106 without it in train set
# If yes, then what features are most important?
# If no, then how do we need to adjust our approach assuming this compound is unique

# Can adding the decoys improve the predictability of the potent compounds?
# If yes, then document and share the steps taken to produce the dataset
# If no, have others had success doing this? Are there more varieties to try?


# Steven
# Predictive model to find anything greater than or less than 50
# Find the optimal number of principal components using visualization/eda

# Next steps
#

>>>>>>> 0276f2ff1a6b6225aa14636b04a1c13b92d2a480
