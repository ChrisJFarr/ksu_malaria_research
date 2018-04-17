import numpy as np


def par_transformations(data, feat):
    data = data.loc[:, feat].copy()
    if data.min() > 0:  # Avoid 0 or negative
        data.loc[:, feat + "_log"] = data[feat].apply(np.log)  # log
        data.loc[:, feat + "_log2"] = data[feat].apply(np.log2)  # log2
        data.loc[:, feat + "_log10"] = data[feat].apply(np.log10)  # log10
    data.loc[:, feat + "_cubert"] = data[feat].apply(
        lambda x: np.power(x, 1 / 3))  # cube root
    data.loc[:, feat + "_sqrt"] = data[feat].apply(np.sqrt)  # square root
    # Avoid extremely large values, keep around 1M max
    if data.max() < 13:
        data.loc[:, feat + "_exp"] = data[feat].apply(np.exp)  # exp
    if data.max() < 20:
        data.loc[:, feat + "_exp2"] = data[feat].apply(np.exp2)  # exp2
    if data.max() < 100:
        data.loc[:, feat + "_cube"] = data[feat].apply(
            lambda x: np.power(x, 3))  # cube
    if data.max() < 1000:
        data.loc[:, feat + "_sq"] = data[feat].apply(np.square)  # square
    return data
