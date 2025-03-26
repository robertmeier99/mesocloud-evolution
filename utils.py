"""
Useful functions for trajectory processing and cloudmetrics computation.
"""

import numpy as np
import xarray as xr
import fsspec
from datetime import datetime

def where_both(condition_1,condition_2):
    return np.where(np.where(condition_1,True,False)*np.where(condition_2,True,False))

def dropna(a):
    return a[np.isfinite(a)]

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
    

def open_GOES_image(datetime):
    """
    Open GOES image closest to given time.
    """
    # open reference dataset with GOES scantimes
    file_path = "~/Data/goes16_reference/goes_ref_ds.nc"
    goes_ref_ds = xr.open_dataset(file_path)

    # check if there is an image close enough
    time_diff = np.min(np.abs(goes_ref_ds.middletime_scan.values-datetime))/10**9
    if time_diff > 10*60:
        print(f"No image +-10 min around {datetime}!")
        return
    
    # find index of closest image
    ref_idx = np.argmin(np.abs(goes_ref_ds.middletime_scan.values-datetime))
    datestring = str(goes_ref_ds.isel(time=ref_idx).datestring.values)
    year = datestring[:4]
    dayoftheyear = convert_datestring_to_dayoftheyear(datestring)
    time_idx = goes_ref_ds.isel(time=ref_idx).t_index.values

    # open image
    file_path = f"/scratch-shared/rmeier/Data/GOES-CMIP-C13-Tropical-North-Atlantic/daily/{year}/OR_ABI-L2-CMIPF-M6C13_G16_{dayoftheyear}.nc"
    CMIPF = xr.open_dataset(file_path).isel(t=time_idx)

    file_path = f"/scratch-shared/rmeier/Data/GOES-ACM-Tropical-North-Atlantic/daily/{year}/OR_ABI-L2-ACMF-M6_G16_{dayoftheyear}.nc"
    ACMF = xr.open_dataset(file_path).isel(t=time_idx)

    return CMIPF, ACMF


def convert_datestring_to_dayoftheyear(datestring):
    return datetime.strptime(datestring,"%Y%m%d").strftime("%j")


def get_centered_window(arr, k):
    """
    Extracts a k x k square window centered at the middle of a 2D array.
    If the window exceeds array bounds, it is NaN-padded.

    Parameters
    ----------
    arr : numpy.ndarray
        Input 2D array of shape (M, N), possibly containing NaNs.
    k : int
        Size of the square window (must be positive).

    Returns
    -------
    window : numpy.ndarray
        Extracted k x k window with NaN-padding if necessary.
    """
    M, N = arr.shape
    i, j = M // 2, N // 2  # Center of the array
    half_k = k // 2

    # Define window boundaries in the original array
    top, bottom = max(0, i - half_k), min(M, i + half_k)
    left, right = max(0, j - half_k), min(N, j + half_k)

    # Compute valid region indices in the output window
    top_pad, bottom_pad = max(0, half_k - i), max(0, (i + half_k + 1) - M - 1)
    left_pad, right_pad = max(0, half_k - j), max(0, (j + half_k + 1) - N - 1)

    # Create a NaN-padded output window
    window = np.full((k, k), np.nan, dtype=arr.dtype)

    # Copy valid region into the output window
    window[top_pad:k-bottom_pad, left_pad:k-right_pad] = arr[top:bottom, left:right]

    return window