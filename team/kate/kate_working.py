import pandas as pd
import numpy as np
df = pd.read_csv("team/kate/column_stats_kate_4.30.csv")

df.dtypes

np.unique(df.categorical, return_counts=True)
np.unique(df.loc[(df.max(axis=1) == 1), "categorical"], return_counts=True)
sum((df.loc[:, "max"] > 1).values & df.categorical.values)

df





