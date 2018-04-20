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

Dimension reduction techniques
1. Load and combine entire dataset including decoys
* Tried PCA with .25% dim len, computer crashed
* Can't perform transformations
* Try nnet to predict target OSM-S-106?
3. Extract feature importance (try different methods)
4. Plot to show potent, non-potent, un-tested, and decoys with top negative dimensions

# Take features selected above and entire dataset
# Train multiple models: hold out the OCM-S-106 as the test set
# Assume all decoys have 100 IC50

Notes:
    * Are there features dropped due to adding decoys that prove as predictors in methods without decoys?
# Try same methods with transforming the IC50 by the method described in.. need to find the paper/website again
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
from par_support import par_transformations
from joblib import Parallel, delayed
from time import time
import par_support
from importlib import reload
reload(par_support)
""" Load and combine all datasets """
file_list = os.listdir("data")
file_list = [f for f in file_list if "decoys" in f and f.endswith(".csv")]

df = pd.read_csv("data/" + file_list[0])
for f in file_list[1:]:
    df = df.append(pd.read_csv("data/" + f))
df["IC50"] = 100
# Combine with Series3_6.15.17_padel.csv
tests = pd.read_csv("data/Series3_6.15.17_padel.csv")
tests.IC50.fillna(-1, inplace=True)  # Mark potential compounds with -1

df = df.append(tests).reset_index(drop=True)
y_data = df.pop("IC50")
x_data = df.dropna(axis=1)
del df
gc.collect()


""" Preprocessing Variables """
# Get dummy vars: filter to int type, convert to object, pass to get_dummies.
assert not sum(x_data[x_data.columns[x_data.iloc[:, :].dtypes == 'int64']].isna().sum()), "Null values found in cat"
cat_vars_df = pd.get_dummies(
    x_data[x_data.columns[x_data.iloc[:, :].dtypes == 'int64']].astype('O'))

# Impute or remove? (for now remove any columns with nan)
cont_vars_df = x_data[x_data.columns[x_data.iloc[:, :].dtypes == 'float64']].dropna(axis=1)

# Assume skewed if we can reject the null hypothesis with 95% certainty
# Remove any skewed features after adding transformations
cont_vars_df = cont_vars_df.loc[:, cont_vars_df.apply(
    lambda x: skewtest(x)[1] > .05).values]

# Combine datasets
x_data = pd.concat([cat_vars_df, cont_vars_df], axis=1)






