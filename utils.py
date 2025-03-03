"""
Useful functions for trajectory processing and cloudmetrics computation.
"""

import numpy as np
import xarray as xr
import fsspec

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

def generate_globsearch_string(
    year, dayofyear, hour=None, channel=None, product="ABI-L2-CMIPF", satellite="goes16"
):
    """
    returns string for glob search in AWS. If hour is not provided, 
    it will download for all files in the given day.
    Function modified from https://github.com/Geet-George/gogoesgone/blob/main/src/gogoesgone/zarr_access.py
    """
    if hour is None:
        return f"s3://noaa-{satellite}/{product}/{year}/{str(dayofyear).zfill(3)}/*/*C{str(channel).zfill(2)}*.nc"
    else:
        if channel is None:
            return f"s3://noaa-{satellite}/{product}/{year}/{str(dayofyear).zfill(3)}/{str(hour).zfill(2)}/*.nc"
        else:
            return f"s3://noaa-{satellite}/{product}/{year}/{str(dayofyear).zfill(3)}/{str(hour).zfill(2)}/*C{str(channel).zfill(2)}*.nc"


def generate_url_list(globsearch_string):
    """
    Returns available URLs' list for AWS filepaths.
    Function copied from https://github.com/Geet-George/gogoesgone/blob/main/src/gogoesgone/zarr_access.py
    """
    fs = fsspec.filesystem("s3", anon=True)
    flist = []
    for f in fs.glob(globsearch_string):
        if f:
            #flist.append("s3://" + f)
            flist = flist + ["s3://" + f]

    if not flist:
        print("No files found!")
        return flist
    else:
        return flist