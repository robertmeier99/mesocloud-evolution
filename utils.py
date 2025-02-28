"""
Useful functions for trajectory processing and cloudmetrics computation.
"""

import numpy as np
import xarray as xr

def where_both(condition_1,condition_2):
    return np.where(np.where(condition_1,True,False)*np.where(condition_2,True,False))

def extract_frame(img,extent):
    w_extent, e_extent, s_extent, n_extent = extent

    return img.where(
            (img.lat >= s_extent)
            & (img.lat <= n_extent)
            & (img.lon >= w_extent)
            & (img.lon <= e_extent),
            drop=True,
        )