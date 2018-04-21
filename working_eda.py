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

# combine = True
combine = False
""" Load and combine all datasets """
if combine:
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
else:
    df = pd.read_csv("data/Series3_6.15.17_padel.csv")
    df.IC50.fillna(-1, inplace=True)  # Mark potential compounds with -1
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
y_data

# Separate untested compounds
untested_i = [i == -1 for i in y_data.values]
full_train_i = [i != -1 for i in y_data.values]
x_untested = x_data.loc[untested_i, :].reset_index(drop=True)
x_data = x_data.loc[full_train_i, :].reset_index(drop=True)
y_untested = y_data.loc[untested_i].reset_index(drop=True)
y_data = y_data.loc[full_train_i].reset_index(drop=True)

# Scale data
x_scaler, y_scaler = StandardScaler(), StandardScaler()
x_norm = x_scaler.fit_transform(x_data)
y_norm = y_scaler.fit_transform(y_data.values.reshape(-1, 1))


# Import models
# Neural nets
# Try architecture varieties: dense, conv, lstm
# Start small
from keras import layers, optimizers, callbacks
from keras import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Goals: predict OSM-S-106 is the best compound yet discovered.
# Use main target as validation
# Create train/validation splits
x_train = x_data.iloc[1:, :].values
y_train = y_data.iloc[1:].values.reshape(-1, 1)
x_valid = x_data.iloc[0, :].values.reshape(-1, x_data.shape[1])
y_valid = np.array(y_data.iloc[0]).reshape(-1, 1)

# Scale inputs
x_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
y_scaler = MinMaxScaler(feature_range=(0.05, 0.99))

x_train_s = x_scaler.fit_transform(x_train)
x_valid_s = x_scaler.transform(x_valid)

# # LSTM (crazy attempt)
x_train_s = x_train_s.reshape(-1, 1, x_data.shape[1])
x_valid_s = x_valid_s.reshape(-1, 1, x_data.shape[1])

y_train_s = y_scaler.fit_transform(y_train)
y_valid_s = y_scaler.transform(y_valid)

# Build simple dense model
d1_in = layers.Input(shape=(1, x_train_s.shape[2]))  # , x_train_s.shape[2]
d1 = layers.LSTM(150)(d1_in)
d1 = layers.BatchNormalization()(d1)
d1 = layers.Activation("relu")(d1)

# d1 = layers.Dropout(0.8)(d1)

d2 = layers.Dense(10)(d1)
d2 = layers.BatchNormalization()(d2)
d2 = layers.Activation("relu")(d2)

# d2 = layers.Dropout(0.8)(d2)

d3 = layers.Dense(1)(d2)
d3 = layers.BatchNormalization()(d3)
d3 = layers.Activation("sigmoid")(d3)

model = Model(inputs=d1_in, outputs=d3)
model.summary()

optimizer = optimizers.Adamax(lr=0.0001)
model.compile(optimizer=optimizer, loss="mse")

# Callbacks
# early_stopping = callbacks.EarlyStopping(patience=10)
# checkpointer = callbacks.ModelCheckpoint("model.best.weight.hdf5", save_weights_only=True, save_best_only=True)

# Train model
# history = model.fit(x_train_s, y_train_s, batch_size=5, epochs=100, callbacks=[early_stopping, checkpointer]
#                     , validation_data=(x_valid_s, y_valid_s), verbose=2)
history = model.fit(x_train_s, y_train_s, steps_per_epoch=1, epochs=10, verbose=2)

# model.load_weights("model.best.weight.hdf5")

# How well can it predict OSM-S-106
print(y_scaler.inverse_transform(model.predict(x_train_s)))
print(y_scaler.inverse_transform(model.predict(x_valid_s)))

# Train MSE
print(np.sqrt(mean_squared_error(y_train, y_scaler.inverse_transform(model.predict(x_train_s)))))
# Potent MSE

# Inverse transform and plot
plt.scatter(y_data, np.squeeze(pred))
plt.xlabel("actual")
plt.ylabel("pred")
plt.show()
# Perform cross validation of all potent compounds when comparing performance



