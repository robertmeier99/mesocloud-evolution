from importlib import reload

import sys
sys.path.append("..")

import os
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import tobac
import time 
import datetime

from gogoesgone.src.gogoesgone import processing as pr
from gogoesgone.src.gogoesgone import zarr_access as za
reload(pr)
reload(za)


def date_and_time(year,day,hour,minutes=None):
    """
    Compute the calendar date and time from hour (with minutes as decimals), day of the year and year input
    """
    # correct for rounding up to 24 h 
    if hour == 24:
        day += 1
        hour = 0
    rest = (hour % 1)
    if minutes==None:
        minutes = str(int(np.rint(rest*60)))
    else:
        minutes = str(minutes)
    minutes = minutes.rjust(2,"0")
    hour = str(int(hour-rest))
    hour = hour.rjust(2,"0")
    time = f"{hour}:{minutes}:00"
    
    date = datetime.datetime.strptime(str(int(year)) + str(int(day)), "%Y%j").strftime("%Y%m%d")
    
    return date, time

def add_date_and_json_index(trajects):
    """
    Adds date and json index to the dataset of trajectories
    """
    N_trajects = trajects.sizes['N_Trajectories']
    

def save_times_array(trajects):
    save_path = "/home/robert/Coding/cloud_org_evolution/ERA_5_traj/Trajectory arrays/"
    start_time = time.time()
    
    # get number of trajectories
    N_trajects = trajects.sizes['N_Trajectories']
    
    for traj_i in range(N_trajects):
        
        traject = trajects.isel(N_Trajectories=traj_i).dropna(dim="Hours_Local_Time")
        traj_number = int(traject.Trajectory_N)
        N_timesteps = traject.sizes['Hours_Local_Time']
        date_array = []
        time_array = []
        json_index_array = np.zeros(N_timesteps,dtype=int)
        
        
        for step in range(N_timesteps):
            # get day and year
            year = int(traject.sel(Hours_Local_Time=step).year_UTC)
            day = float(traject.sel(Hours_Local_Time=step).day_UTC)
            hour = float(traject.sel(Hours_Local_Time=step).hour_UTC)

            # get date and hour
            date, timestamp = date_and_time(year,day,hour)

            # find the closest image
            search_time = f"{date} {timestamp}"
            nearest_url = za.nearest_time_url(search_time, format="%Y%m%d %H:%M:%S", channel=13)
            if nearest_url == "0":
                break
            date_str = nearest_url.split("_e")[0][-14:-3]
            year_found = date_str[:4]
            yday = date_str[4:7]
            hour = date_str[7:9]
            minutes = date_str[9:11]
            # overwrite date for actual image
            date, timestamp = date_and_time(int(year_found),int(yday),int(hour),int(minutes))
            date_array.append(date)
            time_array.append(timestamp)

            # find time index in that days json file
            gs = za.generate_globsearch_string(year_found, yday, channel=13)
            flist = za.generate_url_list(gs)
            json_index_array[step] = flist.index(nearest_url)
        
        if nearest_url == "0":
            print(f"{traj_i} encountered a lack of imagery at {date} {timestamp}.")
            continue
        np.save(f"{save_path}trajectory_{traj_number}_dates",date_array)
        np.save(f"{save_path}trajectory_{traj_number}_times",time_array)
        np.save(f"{save_path}trajectory_{traj_number}_json_indices",json_index_array)
        
        print(f"{traj_i} done after {(time.time()-start_time)/60} min.")
    
    return


def save_center_pos_array(trajects):
    
    save_path = "/home/robert/Coding/cloud_org_evolution/ERA_5_traj/Trajectory arrays/"
    
    # get number of trajectories 
    N_trajects = trajects.sizes['N_Trajectories']
    
    for traj_i in range(N_trajects):
        
        traject = trajects.isel(N_Trajectories=traj_i).dropna(dim="Hours_Local_Time")
        traj_number = int(traject.Trajectory_N)
        
        N_timesteps = traject.sizes['Hours_Local_Time']
        
        center_pos = np.zeros((2,N_timesteps))
        center_pos[0] = traject.longitude
        center_pos[1] = traject.latitude
    
        np.save(f"{save_path}trajectory_{traj_number}_pos", center_pos)
    
    return


def traj_without_missing_image(trajects):
    """
    Find trajectory numbers with missing images
    """
    path = "/home/robert/Coding/cloud_org_evolution/ERA_5_traj/Trajectory arrays/"

    # get trajectory numbers
    traj_numbers = np.unique(trajects.Trajectory_N)
    traj_numbers = traj_numbers.astype(int) 

    # initialize indix list
    with_images = []
    
    for i, traj_number in enumerate(traj_numbers):
        name = f"trajectory_{traj_number}_dates.npy"
        for root, dirs, files in os.walk(path):
            if name in files:
                with_images.append(i)
    
    return trajects.isel(N_Trajectories=with_images)


def where_both(condition_1,condition_2):
    return np.where(np.where(condition_1,True,False)*np.where(condition_2,True,False))

def check_clockwise_rotation(move_lon,move_lat):
    
    clockwise_rot = False
    # rotate to initial SW movement
    # from initial SW movement
    if move_lon[0] < 0 and move_lat[0] < 0:
        move_lon_rot = move_lon
        move_lat_rot = move_lat
    # from initial NW movement
    elif move_lon[0] < 0 and move_lat[0] >= 0:
        move_lon_rot = -move_lat
        move_lat_rot = move_lon
    # from initial NE movement
    elif move_lon[0] >= 0 and move_lat[0] >= 0:
        move_lon_rot = -move_lon
        move_lat_rot = -move_lat
    # from initial SE movement
    else: 
        move_lon_rot = move_lat
        move_lat_rot = -move_lon
        
    # find transition SW - NW
    i_turn = where_both(move_lon_rot < 0, move_lat_rot >= 0)[0]
    if len(i_turn) > 0:
        rest_lon = move_lon_rot[i_turn[0]:]
        rest_lat = move_lat_rot[i_turn[0]:]
        # find transition NW - NE
        i_turn = where_both(rest_lon >= 0, rest_lat >= 0)[0]
        if len(i_turn) > 0:
            rest_lon = rest_lon[i_turn[0]:]
            rest_lat = rest_lat[i_turn[0]:]
            # find transition NE - SE
            i_turn = where_both(rest_lon >= 0, rest_lat < 0)[0]
            if len(i_turn) > 0:
                clockwise_rot = True

    return clockwise_rot

def check_counterclockwise_rotation(move_lon,move_lat):
    
    counterclockwise_rot = False
    # rotate to initial SW movement
    # from initial SW movement
    if move_lon[0] < 0 and move_lat[0] < 0:
        move_lon_rot = move_lon
        move_lat_rot = move_lat
    # from initial NW movement
    elif move_lon[0] < 0 and move_lat[0] >= 0:
        move_lon_rot = -move_lat
        move_lat_rot = move_lon
    # from initial NE movement
    elif move_lon[0] >= 0 and move_lat[0] >= 0:
        move_lon_rot = -move_lon
        move_lat_rot = -move_lat
    # from initial SE movement
    else: 
        move_lon_rot = move_lat
        move_lat_rot = -move_lon
        
    # find transition SW - SE
    i_turn = where_both(move_lon_rot >= 0, move_lat_rot < 0)[0]
    if len(i_turn) > 0:
        rest_lon = move_lon_rot[i_turn[0]:]
        rest_lat = move_lat_rot[i_turn[0]:]
        # find transition SE - NE
        i_turn = where_both(rest_lon >= 0, rest_lat >= 0)[0]
        if len(i_turn) > 0:
            rest_lon = rest_lon[i_turn[0]:]
            rest_lat = rest_lat[i_turn[0]:]
            # find transition NE - NW
            i_turn = where_both(rest_lon < 0, rest_lat >= 0)[0]
            if len(i_turn) > 0:
                counterclockwise_rot = True

    return counterclockwise_rot


def filter_out_loops(trajects):
    """
    Filters out trajectories with loops
    """
    # get number of trajectories 
    N_trajects = trajects.sizes['N_Trajectories']
    no_loops = []
    loops_cw = []
    loops_ccw = []
    
    for i in range(N_trajects):
        traject = trajects.isel(N_Trajectories=i).dropna(dim="Hours_Local_Time",how="all")
        
        # get lat lon movement
        lat = np.array(traject.latitude)
        lon = np.array(traject.longitude)
        move_lat = lat[1:] - lat[:-1]
        move_lon = lon[1:] - lon[:-1]
        
        clockwise_rot = check_clockwise_rotation(move_lon,move_lat)
        
                    
        if clockwise_rot:
            loops_cw.append(i)
        else:
            counterclockwise_rot = check_counterclockwise_rotation(move_lon,move_lat)
            if counterclockwise_rot:
                loops_ccw.append(i)
            else:
                no_loops.append(i)
                
    return trajects.isel(N_Trajectories=no_loops), trajects.isel(N_Trajectories=loops_cw), trajects.isel(N_Trajectories=loops_ccw)


def map_tracks(track, axis_extent=None, figsize=(10,8), dpi=100, untracked_cell_value=-1):
    """Plot the trajectories of the cells on a map.

    Parameters
    ----------
    track : xarray.Dataset
        Dataset containing the trajectories

    axis_extent : matplotlib.axes, optional
        Array containing the bounds of the longitude
        and latitude values. The structure is
        [long_min, long_max, lat_min, lat_max].
        Default is None.

    figsize : tuple of floats, optional
        Width, height of the plot in inches.
        Default is (10, 10).

    untracked_cell_value : int or np.nan, optional
        Value of untracked cells in track['cell'].
        Default is -1.

    Raises
    ------
    ValueError
        If no axes is passed.
    """

    fig_map,axes = plt.subplots(figsize=figsize,dpi=dpi,subplot_kw={'projection': ccrs.PlateCarree()})
    
    if axes is None:
        raise ValueError(
            "axes needed to plot tracks onto. Pass in an axis to axes to resolve this error."
        )
    traject_num = np.array(track.Trajectory_N)
    for cell in traject_num:
        if cell == untracked_cell_value:
            continue
        track_i = track.isel(N_Trajectories=np.where(traject_num == cell)[0]).dropna(dim="Time")
        starttime = pd.to_datetime(np.min(track_i.datetime_UTC.values)).strftime("%d-%m-%Y %H:%M")
        endtime = pd.to_datetime(np.max(track_i.datetime_UTC.values)).strftime("%d-%m-%Y %H:%M")

        axes.plot(np.array(track_i.longitude), np.array(track_i.latitude), "-", label=f"From {starttime} to {endtime}")

        if axis_extent:
            axes.set_extent(axis_extent)
        axes = tobac.make_map(axes)

    axes.legend()

    return


def add_datetime(ds):
    """
    Add datetime DataArray to Dataset of trajectories
    """
    # get UTC time inputs
    years = ds.year_UTC.values
    days = ds.day_UTC.values
    hours = ds.hour_UTC.values
    
    # correct for rounding up to 24 h 
    days[np.where(hours==24)] += 1
    hours[np.where(hours==24)] = 0
    
    # get minutes and seconds 
    minutes_decimal = hours % 1
    minutes = minutes_decimal*60
    seconds_decimal = minutes % 1
    seconds = np.rint(seconds_decimal*60).astype(int)
    minutes = (minutes - seconds_decimal).astype(int)
    hours = (hours - minutes_decimal).astype(int)
    
    # correct for 00:00:60
    minutes[np.where(seconds==60)] += 1
    seconds[np.where(seconds==60)] = 0 
    
    # compute datetimes
    date_time = np.empty(np.shape(years),dtype=datetime.datetime)
    for i in range(np.shape(years)[0]):
        for j in range(np.shape(years)[1]):
            if np.isnan(years[i,j]) == False:
                date = datetime.datetime.strptime(str(int(years[i,j])) + str(int(days[i,j])), "%Y%j")
                date_time[i,j] = datetime.datetime(date.year,date.month,date.day,hours[i,j],minutes[i,j],seconds[i,j])
                
    # add datetimes to Dataset
    da = xr.DataArray(data=date_time,dims=["Hours_Local_Time","N_Trajectories"])
    ds = ds.assign(datetime_UTC = da)
    
    return ds


def interpolate_trajects(trajects,goes_ref_ds,N_timesteps=960):
    """
    Interpolates trajectories linearly from 1 hourly trajectories onto the 10/15-min GOES images.
    """
    
    # Get trajectory numbers and amount of them
    Trajectory_N = trajects.Trajectory_N.values
    N_Trajectories = len(Trajectory_N)
    
    # Initialize arrays to build new interpolated Dataset
    longitudes = np.full((N_timesteps,N_Trajectories),np.nan)
    latitudes = np.full((N_timesteps,N_Trajectories),np.nan)
    datetime_UTC = np.full((N_timesteps,N_Trajectories),np.nan).astype("datetime64[ns]")
    
    # get central time of each GOES scan
    scan_starts = goes_ref_ds.starttime_scan
    scan_ends = goes_ref_ds.endtime_scan
    scan_middle_time = scan_starts + (scan_ends-scan_starts)/2
    
    for i in range(N_Trajectories):
        # select trajectory 
        track_i = trajects.isel(N_Trajectories=i).dropna(dim="Hours_Local_Time")   # Note: by dropping NaNs here we lose alignment by local hour
        
        # longitude cut-off at 60°W
        over_ocean = np.where(track_i.longitude > -60)[0]
        
        # get start and endtime of trajectory above the Atlantic
        traj_times = track_i.isel(Hours_Local_Time=over_ocean).datetime_UTC
        starttime = np.min(traj_times)
        endtime = np.max(traj_times)
        
        # get scan times that are in between start and end of trajectory
        traj_img_times = scan_middle_time[where_both(scan_middle_time>starttime,scan_middle_time<endtime)]
        datetime_UTC[:len(traj_img_times),i] = traj_img_times
       
        # interpolate lat and lon along these image times
        for j, traj_img_time in enumerate(traj_img_times):
            time_diffs = (traj_img_time - traj_times).astype(int)
            
            # get temporal difference to predecessor and successor trajectory point
            m = np.min(time_diffs[np.where(time_diffs>0)])
            n = -np.max(time_diffs[np.where(time_diffs<0)])
            
            # get predecessor and successor trajectory point indices
            past_traj_ind = np.where(time_diffs==m)[0][0]
            fut_traj_ind = np.where(time_diffs==-n)[0][0]
            
            # interpolate
            longitudes[j,i] = track_i.longitude[past_traj_ind] + m/(m+n)*(track_i.longitude[fut_traj_ind]-track_i.longitude[past_traj_ind])
            latitudes[j,i] = track_i.latitude[past_traj_ind] + m/(m+n)*(track_i.latitude[fut_traj_ind]-track_i.latitude[past_traj_ind])

    ds = xr.Dataset(data_vars=dict(Trajectory_N=(["N_Trajectories"],Trajectory_N),
                                    longitude=(["Time","N_Trajectories"],longitudes),
                                    latitude=(["Time","N_Trajectories"],latitudes),
                                    datetime_UTC=(["Time","N_Trajectories"],datetime_UTC)),
                      attrs=dict(description="Trajectory data interpolated on GOES images"))

    return ds.dropna(dim="Time",how="all")

