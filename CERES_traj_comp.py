"""
Calculation of CERES averages along interpolated ERA5 trajectories.
"""

import numpy as np
import xarray as xr
import time 
import glob
import os
import gc
import resource
from line_profiler import LineProfiler


def main():
    # set inputs
    remote = False
    area_weighted = False
    save_freq = 1000 # save after every 1000 trajectories
    framesize = 5
    profiler = False

    if remote:
        data_path = "/scratch/rmeier1/Data/CERES/CERES_DJF_17-22.nc"
        traj_dir = "/scratch/rmeier1/Data/Trajectories/"
        save_dir = "/scratch/rmeier1/Data/Trajectories/"
        traj_file_name = "NAtl_Trajectories_Mid_Start_925hPa_CERES_interp_Dec-Feb_2017-2022"
    else:
        data_path = "/home/rmeier1/PhD/Datasets/CERES/CERES_DJF_17-22.nc"
        traj_dir = "/home/rmeier1/PhD/Datasets/interp_data/"
        save_dir = "/home/rmeier1/PhD/Datasets/CERES/"
        traj_file_name = "NAtl_Trajectories_Mid_Start_925hPa_CERES_interp_Dec-Feb_2017-2022"

    start = time.time()

    if profiler:
        #initialize line_profiler
        lp = LineProfiler()
        lp_wrapper = lp(compute_CERES_along_traj)

    # get datasets
    CERES_ds = xr.open_dataset(data_path)
    trajects = xr.open_dataset(traj_dir + traj_file_name + ".nc")

    # define which variables to average
    mean_vars = list(CERES_ds)[:-2]

    print("Start computation...")
    if profiler:
        lp_wrapper(trajects,CERES_ds,mean_vars,area_weighted,framesize,save_dir,traj_file_name,save_freq)
        lp.print_stats()
    else:
        compute_CERES_along_traj(trajects,CERES_ds,mean_vars,area_weighted,framesize,save_dir,traj_file_name,save_freq)
 
    print("programm completed in" + str(round(time.time()-start,0)) + "s.")
    print("memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "Kb")


def compute_CERES_along_traj(traj_ds,CERES_ds,mean_vars,area_weighted,framesize,save_dir,traj_file_name,save_freq=1000):
    """
    Compute CERES variable averages over frames centered along trajectory locations.

    Parameters
    ---------------------------------------------------------------------------------------
    - traj_ds: Dataset of trajectories interpolated onto CERES times
    - CERES_ds: Dataset of hourly CERES variables 
    - mean_vars: List of variables to average 
    - area_weighted: boolean, True to apply area weighted averages 
    - framesize: size of the frame to average over 
    - save_dir: directory or path to save files at
    - traj_file_name: file_name of trajectories, used with appendix for saving output
    - save_freq: number of trajectories after which output is saved (default 1000)

    Return
    ---------------------------------------------------------------------------------------
    - CERES_traj: Trajectories with assigned averages of CERES variables 
    """

    N_Trajectories = traj_ds.sizes["N_Trajectories"]
    N_Timesteps = traj_ds.sizes["Time"]
    mean_vars = list(CERES_ds)[:-2]
    
    # initialize CERES variables along trajectory
    CERES_traj = {}
    for var in mean_vars:
        CERES_traj[var + "_spatial_mean"] = np.full((N_Timesteps,N_Trajectories),np.nan)

    for i in range(N_Trajectories):
        print(f"{np.round(i/N_Trajectories*100)} % complete",end="\r")
        traj = traj_ds.isel(N_Trajectories=i)

        traj_time = traj.datetime_UTC.values
        traj_lon = traj.longitude.values
        traj_lat = traj.latitude.values

        # load seasonal subset of CERES data
        if i==0:
            min_time = np.nanmin(traj_time)
            max_time = np.nanmax(traj_time) + np.timedelta64(150,"D")
            CERES_sub = CERES_ds.sel(time=slice(min_time,max_time)).load()
        elif np.nanmax(traj_time) > max_time:
            # Delete old subset
            del CERES_sub
            gc.collect()
            # Load new subset
            min_time = np.nanmin(traj_time)
            max_time = np.nanmax(traj_time) + np.timedelta64(150,"D")
            CERES_sub = CERES_ds.sel(time=slice(min_time,max_time)).load()
        elif np.nanmin(traj_time) < min_time:
            # Delete old subset
            del CERES_sub
            gc.collect()
            # Load new subset
            min_time = np.nanmin(traj_time)
            max_time = np.nanmax(traj_time) + np.timedelta64(150,"D")
            CERES_sub = CERES_ds.sel(time=slice(min_time,max_time)).load()

        for j in range(len(traj_time)):

            # leave NaN, if 5x5 frame is outside of the data region
            if (traj_lon[j] < -67.5)+(traj_lon[j] > -12.5)+(traj_lat[j] < 2.5)+(traj_lat[j] > 37.5) > 0:
                continue

            CERES_frame = CERES_sub.sel(time=traj_time[j], method="nearest").sel(lon=slice(traj_lon[j]-framesize/2, traj_lon[j]+framesize/2), 
                                                                                lat=slice(traj_lat[j]-framesize/2, traj_lat[j]+framesize/2))
            
            if len(mean_vars) > 0:
                if area_weighted:
                    CERES_mean = area_weighted_average(CERES_frame,mean_vars)
                else:
                    CERES_mean = {
                                    var: CERES_frame[var].mean(dim=('lat', 'lon'), skipna=True)
                                    for var in mean_vars
                                }

                for var in mean_vars:
                    CERES_traj[var + "_spatial_mean"][j,i] = CERES_mean[var]

        # intermediate output
        if (i>0)*(i%save_freq==0):
            CERES_traj_dim = {}
            for var in mean_vars:
                CERES_traj_dim[var + "_spatial_mean"] = (["Time","N_Trajectories"],CERES_traj[var + "_spatial_mean"])

            if len(glob.glob(save_dir + traj_file_name + "_with_CERES.nc")) > 0:
                os.remove(save_dir + traj_file_name +"_with_CERES.nc")

            print("Saving...")
            traj_ds.assign(variables=CERES_traj_dim).to_netcdf(save_dir + traj_file_name +"_with_CERES.nc")

    # final output
    CERES_traj_dim = {}
    for var in mean_vars:
        CERES_traj_dim[var + "_spatial_mean"] = (["Time","N_Trajectories"],CERES_traj[var + "_spatial_mean"])

    if len(glob.glob(save_dir + traj_file_name + "_with_CERES.nc")) > 0:
        os.remove(save_dir + traj_file_name +"_with_CERES.nc")

    traj_ds.assign(variables=CERES_traj_dim).to_netcdf(save_dir + traj_file_name +"_with_CERES.nc")

    return 


def area_weighted_average(ds,vars):
    w = np.cos(np.deg2rad(ds.lat)) 
    w /= w.sum()
    return ds[vars].weighted(w).mean(dim=["lat","lon"],skipna=True)


if __name__ == "__main__":
    main()
