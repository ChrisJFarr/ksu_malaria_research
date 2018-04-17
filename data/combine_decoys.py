import os
import pandas as pd
import gc

file_list = os.listdir("data")
file_list = [f for f in file_list if "decoys" in f and f.endswith(".csv")]

df = pd.read_csv("data/" + file_list[0])
for f in file_list[1:]:
    df = df.append(pd.read_csv("data/" + f))

# Combine with Series3_6.15.17_padel.csv
tests = pd.read_csv("data/Series3_6.15.17_padel.csv")
tests.IC50.fillna(-1, inplace=True)

df = df.append(tests).reset_index(drop=True)
y_data = df.pop("IC50")
x_data = df.dropna(axis=1)
del df
gc.collect()

# Primarily floats have complete features across datasets
print("%s continuous features" % sum(x_data.dtypes == "float64"))
print("%s categorical features" % sum(x_data.dtypes == "int64"))



