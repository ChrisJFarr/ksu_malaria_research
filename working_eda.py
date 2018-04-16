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
from sklearn.linear_model import Lasso
from scipy.stats import skewtest

df = pd.read_csv("data/Series3_6.15.17_padel.csv")
# Drop examples without IC50
df = df[~df.IC50.isnull()].copy()

# Column types and counts
np.unique(df.dtypes)
len(df.columns[df.iloc[:, :].dtypes == 'O'])
len(df.columns[df.iloc[:, :].dtypes == 'int64'])
len(df.columns[df.iloc[:, :].dtypes == 'float64'])

""" Preprocessing Variables """
# Categorical Variables: No missing values
sum(df[df.columns[df.iloc[:, :].dtypes == 'int64']].isnull().sum())
# Get dummy vars: filter to int type, convert to object, pass to get_dummies.
cat_vars_df = pd.get_dummies(df[df.columns[df.iloc[:, :].dtypes == 'int64']].astype('O'))

# Continuous Variables: 67 columns have missing values
sum(df[df.columns[df.iloc[:, :].dtypes == 'float64']].isnull().sum())
# Impute or remove? (for now remove any columns with nan
cont_vars_df = df[df.columns[df.iloc[:, :].dtypes == 'float64']].dropna(axis=1)
# Drop target variablea
cont_vars_df.drop("IC50", axis=1, inplace=True)


def add_transformations(df, feat):
    feature_df = df.loc[:, feat].copy()
    if feature_df.min() > 0:
        df.loc[:, feat + "_log"] = feature_df.apply(np.log)  # log
        df.loc[:, feat + "_log2"] = feature_df.apply(np.log2)  # log2
        df.loc[:, feat + "_log10"] = feature_df.apply(np.log10)  # log10
    df.loc[:, feat + "_exp"] = feature_df.apply(np.exp)  # exp
    df.loc[:, feat + "_exp2"] = feature_df.apply(np.exp2)  # exp2
    df.loc[:, feat + "_cubert"] = feature_df.apply(lambda x: np.power(x, 1/3))  # cube root
    df.loc[:, feat + "_sqrt"] = feature_df.apply(np.sqrt)  # square root
    if feature_df.max() < 10:
        df.loc[:, feat + "_sq"] = feature_df.apply(np.square)  # square
        df.loc[:, feat + "_cube"] = feature_df.apply(lambda x: np.power(x, 3))  # cube
    return df


for feature in cont_vars_df.columns:
    cont_vars_df = add_transformations(cont_vars_df, feature)
# Drop any new columns with NaN due to improper transformation
cont_vars_df.replace([np.inf, -np.inf], np.nan, inplace=True)
cont_vars_df.dropna(axis=1, inplace=True)

# Assume skewed if we can reject the null hypothesis with 95% certainty
# Remove any skewed features after adding transformations
cont_vars_df = cont_vars_df.loc[:, cont_vars_df.apply(lambda x: skewtest(x)[1] > .05).values]
cont_vars_df.columns
# Combine datasets
vars_df = pd.concat([cat_vars_df, cont_vars_df], axis=1)


# Feature interactions
# log of ratio

# Normalize variables
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_train = x_scaler.fit_transform(vars_df)
x_columns = list(vars_df.columns)
y_train = y_scaler.fit_transform(df.IC50.values.reshape(-1, 1))

# Forward stepwise feature selection, with the main target

# Forward stepwise feature selection, with target as validation
# From what we knew prior


# Build model
# NOTE: Gridsearch with Lasso didn't work well, "best" model output all 0's for coefficients
model = Lasso(alpha=0.25, max_iter=100000, tol=1e-5)
model.fit(x_train, y_train)

# # TODO perform loocv to generate prediction
# prediction = model.predict(x_train)
# from sklearn.metrics import mean_squared_error
# mean_squared_error(y_train, prediction)
# pd.DataFrame({"actual": list(np.squeeze(y_train)), "pred": list(prediction)})

# Extract coefficients
positive_coeffs = len([c for c in model.coef_ if c > 0])
neg_coeffs = len([c for c in model.coef_ if c < 0])
# Negative predictive indicators are selected, potency-driving features
pred_indicators = [f for f, c in zip(x_columns, model.coef_) if c != 0]
print(len(pred_indicators))
""" PCA with selected features """


# Manually separate color map to focus on effective compounds
# If less 10 make zero, target -.5, else 1
# TODO we could try different values here
def create_map(x):
    if x < 1:
        return 'black'
    elif x < 10:
        return 'blue'
    elif x < 15:
        return 'deepskyblue'
    else:
        return 'lightgreen'


color_map = df.IC50.apply(create_map)

# Build Linear Pricipal Components
pca = KernelPCA(n_components=2, kernel="poly", random_state=0, n_jobs=3)
pca_out = pca.fit_transform(vars_df.loc[:, pred_indicators])
# Sort pca so potent show on top of non-potent (easier visibility)
sorted_pca = [x for (y, x) in sorted(zip(df.IC50.values, pca_out), key=lambda pair: pair[0], reverse=True)]
color_map = [x for (y, x) in sorted(zip(df.IC50.values, color_map), key=lambda pair: pair[0], reverse=True)]
sorted_pca = np.array(sorted_pca)
x = sorted_pca[:, 0]
y = sorted_pca[:, 1]
plt.scatter(x=x, y=y, c=color_map, alpha=1)
plt.xlim(min(x) * 1.1, max(x) * 1.1)
plt.ylim(min(y) * 1.1, max(y) * 1.1)
plt.title("Polynomial PCA")
plt.show()

