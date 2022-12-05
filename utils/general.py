"""
Define utility function for general use.
"""
import numpy as np


def safe_wma(vals, wghts):
    """
    Weighted moving average, which does not throw error when applied on the empty list and does not take Nans
    into account.
    """
    if vals.shape[0] == 0 or wghts.sum() == 0:
        return np.nan
    else:
        return np.average(vals[~np.isnan(vals)], weights=wghts[~np.isnan(vals)])