"""
-------------------------------------------------------------------------------------------
Trajectory preprocessing functions for filtering and interpolating ERA5 trajectories.
-------------------------------------------------------------------------------------------
"""

import sys
sys.path.append("..")

import numpy as np
import xarray as xr
from datetime import datetime,timedelta

from utils import where_both, generate_globsearch_string, generate_url_list


def filter_out_loops(trajects):
    """
    Filters out trajectories with loops.

    Input:
    ---------------------------------------------------------------------------------------
    - trajects    Dataset with trajectories (dimensions: N_Trajectories, Hours_Local_Time)

    Output:
    ---------------------------------------------------------------------------------------
    - trajects_loop_free    Dataset containing trajectories without loops
    - trajects_loop_cw      Dataset containing trajectories with clockwise loops
    - trajects_loop_ccw     Dataset containing trajectories with counterclockwise loops
    """
    
    # Initialize index lists
    no_loops_idx = []
    cw_loops_idx = []
    ccw_loops_idx = []
    
    N_trajects = trajects.sizes['N_Trajectories']
    
    for i in range(N_trajects):
        traject = trajects.isel(N_Trajectories=i).dropna(dim="Hours_Local_Time",how="all")
        
        # get lat lon tendencies
        lat = traject.latitude.values
        lon = traject.longitude.values
        lat_tendency = np.diff(lat)
        lon_tendency = np.diff(lon)
        
        # Sort trajectory index in one of the index lists
        has_cw_rot = check_clockwise_rotation(lon_tendency,lat_tendency)
                    
        if has_cw_rot:
            cw_loops_idx.append(i)
        else:
            has_ccw_rot = check_counterclockwise_rotation(lon_tendency,lat_tendency)

            if has_ccw_rot:
                ccw_loops_idx.append(i)
            else:
                no_loops_idx.append(i)
                
    # filter trajectories by index
    trajects_loop_free = trajects.isel(N_Trajectories=no_loops_idx)
    trajects_loop_cw = trajects.isel(N_Trajectories=cw_loops_idx)
    trajects_loop_ccw = trajects.isel(N_Trajectories=ccw_loops_idx)

    return trajects_loop_free, trajects_loop_cw, trajects_loop_ccw


def check_clockwise_rotation(dx,dy):
    """
    Check if a clockwise rotation occurs from tendencies in 2D trajectory.

    Input:
    ---------------------------------------------------------------------------------------
    - dx    Array of tendencies in x-direction 
    - dy    Array of tendencies in y-direction

    Output:
    ---------------------------------------------------------------------------------------
    - has_cw_rot    True if trajectory has clockwise rotation, False if not
    """
    # initialize output
    has_cw_rot = False

    # rotate to initial SW movement
    # from initial SW movement
    if dx[0] < 0 and dy[0] < 0:
        dx_rot = dx
        dy_rot = dy
    # from initial NW movement
    elif dx[0] < 0 and dy[0] >= 0:
        dx_rot = -dy
        dy_rot = dx
    # from initial NE movement
    elif dx[0] >= 0 and dy[0] >= 0:
        dx_rot = -dx
        dy_rot = -dy
    # from initial SE movement
    else: 
        dx_rot = dy
        dy_rot = -dx
        
    # find transition SW - NW
    i_turn = where_both(dx_rot < 0, dy_rot >= 0)[0]
    if len(i_turn) > 0:
        rest_lon = dx_rot[i_turn[0]:]
        rest_lat = dy_rot[i_turn[0]:]
        # find transition NW - NE
        i_turn = where_both(rest_lon >= 0, rest_lat >= 0)[0]
        if len(i_turn) > 0:
            rest_lon = rest_lon[i_turn[0]:]
            rest_lat = rest_lat[i_turn[0]:]
            # find transition NE - SE
            i_turn = where_both(rest_lon >= 0, rest_lat < 0)[0]
            if len(i_turn) > 0:
                has_cw_rot = True

    return has_cw_rot


def check_counterclockwise_rotation(dx,dy):
    """
    Check if a counterclockwise rotation occurs from tendencies in 2D trajectory.

    Input:
    ---------------------------------------------------------------------------------------
    - dx    Array of tendencies in x-direction 
    - dy    Array of tendencies in y-direction

    Output:
    ---------------------------------------------------------------------------------------
    - has_ccw_rot    True if trajectory has counterclockwise rotation, False if not
    """
    has_ccw_rot = False
    # rotate to initial SW movement
    # from initial SW movement
    if dx[0] < 0 and dy[0] < 0:
        dx_rot = dx
        dy_rot = dy
    # from initial NW movement
    elif dx[0] < 0 and dy[0] >= 0:
        dx_rot = -dy
        dy_rot = dx
    # from initial NE movement
    elif dx[0] >= 0 and dy[0] >= 0:
        dx_rot = -dx
        dy_rot = -dy
    # from initial SE movement
    else: 
        dx_rot = dy
        dy_rot = -dx
        
    # find transition SW - SE
    i_turn = where_both(dx_rot >= 0, dy_rot < 0)[0]
    if len(i_turn) > 0:
        rest_lon = dx_rot[i_turn[0]:]
        rest_lat = dy_rot[i_turn[0]:]
        # find transition SE - NE
        i_turn = where_both(rest_lon >= 0, rest_lat >= 0)[0]
        if len(i_turn) > 0:
            rest_lon = rest_lon[i_turn[0]:]
            rest_lat = rest_lat[i_turn[0]:]
            # find transition NE - NW
            i_turn = where_both(rest_lon < 0, rest_lat >= 0)[0]
            if len(i_turn) > 0:
                has_ccw_rot = True

    return has_ccw_rot

def add_datetime(ds):
    """
    Add UTC datetime DataArray to Dataset of trajectories.
    """
    # get UTC time inputs
    years = ds.year_UTC.values
    days = ds.day_UTC.values
    hours = ds.hour_UTC.values
    
    # compute minutes and seconds 
    hours, mins, secs = compute_h_min_sec_from_decimals(hours)
    
    # set 24:00 to 0:00 the next day and add year (if necessary)
    years, days, hours, mins, secs = corr_t_round_err(years,days,hours,mins,secs) 
    
    # compute datetimes
    date_time = np.empty(np.shape(years),dtype=datetime)

    for i in range(np.shape(years)[0]):
        for j in range(np.shape(years)[1]):

            if np.isfinite(years[i,j]):
                date_time[i,j] = datetime.strptime(str(int(years[i,j])) 
                                                + str(int(days[i,j]))
                                                + "T" + str(hours[i,j])
                                                + ":" + str(mins[i,j])
                                                + ":" + str(secs[i,j]), "%Y%jT%H:%M:%S")
                
    # add datetimes to Dataset
    da = xr.DataArray(data=date_time,dims=["Hours_Local_Time","N_Trajectories"])
    ds = ds.assign(datetime_UTC = da)
    
    return ds

def compute_h_min_sec_from_decimals(decimal_hours):
    """
    Compute integer hours, minutes and seconds from decimal hours. 
    """
    minutes_decimal = decimal_hours % 1
    minutes = minutes_decimal*60
    seconds_decimal = minutes % 1
    seconds = np.rint(seconds_decimal*60).astype(int)
    minutes = (minutes - seconds_decimal).astype(int)
    hours = (decimal_hours - minutes_decimal).astype(int)

    return hours, minutes, seconds

def corr_t_round_err(years,days,hours,mins,secs):
    """
    Correct for rounding errors like mins = 60, hours = 24 or days = 366 (non-leap years).
    """
    # minute jump
    mins[secs>=60] += 1
    secs = np.where(secs>=60,secs % 60,secs)
    # hour jump
    hours[mins>=60] += 1
    mins = np.where(mins>=60,mins % 60,mins)
    # day jump
    days[hours>=24] += 1
    hours = np.where(hours>=24,hours % 24,hours)
    # year jump
    year_jump = ((years%4)>0)*(days==366) + ((years%4)==0)*(days==367)
    years[year_jump] += 1
    days[year_jump] = 1

    return years, days, hours, mins, secs

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
    scan_middle_times = goes_ref_ds.middletime_scan.values
    
    for i in range(N_Trajectories):
        # select trajectory 
        traject = trajects.isel(N_Trajectories=i)#.dropna(dim="Hours_Local_Time")   # Note: by dropping NaNs here we lose alignment by local hour
        
        # get trajectory times and locations
        traj_times = traject.datetime_UTC.values
        traj_lons = traject.longitude.values
        traj_lats = traject.latitude.values

        # interpolate trajectory onto GOES scantimes
        traj_interp = temp_interp_2D(traj_times,traj_lons,traj_lats,scan_middle_times)

        datetime_UTC[:len(traj_interp[0]),i] = traj_interp[0]
        longitudes[:len(traj_interp[1]),i] = traj_interp[1]
        latitudes[:len(traj_interp[2]),i] = traj_interp[2]

    ds = xr.Dataset(data_vars=dict(Trajectory_N=(["N_Trajectories"],Trajectory_N),
                                    longitude=(["Time","N_Trajectories"],longitudes),
                                    latitude=(["Time","N_Trajectories"],latitudes),
                                    datetime_UTC=(["Time","N_Trajectories"],datetime_UTC)),
                      attrs=dict(description="Trajectory data interpolated on GOES images"))

    return ds.dropna(dim="Time",how="all")

def temp_interp_2D(t,x,y,t_hr):
    """
    Interpolate 2D trajectory locations temporally. First checks which of the high 
    resolution times are within the time range of the trajectory.

    Input:
    ---------------------------------------------------------------------------------------
    - t:        time coordinate of the input trajectory
    - x:        x-coordinate of the input trajectory
    - y:        y-coordinate of the input trajectory
    - t_hr:     higher resoultion time coordinate 

    Output:
    ---------------------------------------------------------------------------------------
    - t_interp: time coordinate of interpolated trajectory
    - x_interp: x-coordinate of interpolated trajectory
    - y_interp: y-coordinate of interpolated trajectory
    """
    # high res times within the trajectory time range
    t_interp = t_hr[(t_hr > np.min(t))*(t_hr < np.max(t))]

    # initialize output
    x_interp = np.empty(len(t_interp))
    y_interp = np.empty(len(t_interp))

    for j in range(len(t_interp)):
        delta_t = (t-t_interp[j]).astype(int)

        # get temporal difference to predecessor and successor trajectory point
        delta_t_next = np.min(delta_t[delta_t>=0])
        delta_t_prev = -np.max(delta_t[delta_t<0])
        
        # get predecessor and successor trajectory point indices
        next_traj_idx = np.where(delta_t==delta_t_next)[0][0]
        prev_traj_idx = np.where(delta_t==-delta_t_prev)[0][0]
        
        # interpolate
        interp_factor = delta_t_prev/(delta_t_prev+delta_t_next)
        x_interp[j] = x[prev_traj_idx] + interp_factor*(x[next_traj_idx]-x[prev_traj_idx])
        y_interp[j] = y[prev_traj_idx] + interp_factor*(y[next_traj_idx]-y[prev_traj_idx])
    
    return t_interp, x_interp, y_interp

def get_goes_ref_ds(years,months,margin=4):
    """
    Generate Dataset that stores GOES-16 scantimes (start,middle,end), datestrings
    related to the daily netCDF/json files and indices for the time of the day.
    
    Input:
    ---------------------------------------------------------------------------------------
    - years:        List of strings giving the years of interest
    - months:       List of integers giving the months of interest
    - margin:       Integer number of days before and after seasonal period of interest
                    (default 4 for 6-day trajectories)

    Output:
    ---------------------------------------------------------------------------------------
    - goes_ref_ds   Dataset of reference to GOES-16 images available from AWS storage
    """
    # generate list of strings with zeropadded days of year 
    daysofyear = np.arange(1,367).astype(str)
    for i in range(len(daysofyear)):
        daysofyear[i] = daysofyear[i].rjust(3,"0")

    # take out days within given months +- margin of days
    if np.any(np.array(years).astype(int)%4 == 0):
        date_in_margin = np.zeros(366,dtype="bool")
        for i,dayofyear in enumerate(daysofyear):
            date = datetime.strptime("2020"+dayofyear,"%Y%j") 
            date_in_margin[i] = ((date + timedelta(margin)).month in months) or ((date - timedelta(margin)).month in months)
        daysofleapyear = daysofyear[date_in_margin]
    if np.any(np.array(years).astype(int)%4 > 0):
        date_in_margin = np.zeros(366,dtype="bool")
        for i,dayofyear in enumerate(daysofyear[:-1]):
            date = datetime.strptime("2019"+dayofyear,"%Y%j") 
            date_in_margin[i] = ((date + timedelta(margin)).month in months) or ((date - timedelta(margin)).month in months)
        daysofnonleapyear = daysofyear[date_in_margin]
    
    # initialize output
    max_length = (len(years)*len(months)*31 + 2*margin)*24*6
    scan_start = np.empty(max_length,dtype="datetime64[ns]")
    scan_end = np.empty(max_length,dtype="datetime64[ns]")
    datestr = np.empty(max_length)
    time_ind = np.empty(max_length)
    counter_idx = 0

    # GOES data naming convention
    time_format = "%Y%j%H%M%S"

    for year in years:
        if int(year)%4 == 0:        # leap years

            for dayofyear in daysofleapyear:
                print(year + " " + dayofyear)
                gss = generate_globsearch_string(year,dayofyear,channel=13, product="ABI-L2-CMIPF", satellite="goes16")
                flist = generate_url_list(gss)
                if len(flist) == 0:
                    continue
                for i in range(len(flist)):
                    scan_s = datetime.strptime(flist[i].split("_e")[0].split("_s")[1][:-1],time_format)
                    scan_e = datetime.strptime(flist[i].split("_e")[1].split("_c")[0][:-1],time_format)
                    scan_start[counter_idx] = scan_s
                    scan_end[counter_idx] = scan_e
                    datestr[counter_idx] = datetime.strftime(scan_s,"%Y%m%d")
                    time_ind[counter_idx] = i
                    counter_idx =+ 1

        else:                       # non-leap years

            for dayofyear in daysofnonleapyear:
                print(year + " " + dayofyear)
                gss = generate_globsearch_string(year,dayofyear,channel=13, product="ABI-L2-CMIPF", satellite="goes16")
                flist = generate_url_list(gss)
                if len(flist) == 0:
                    continue
                for i in range(len(flist)):
                    scan_s = datetime.strptime(flist[i].split("_e")[0].split("_s")[1][:-1],time_format)
                    scan_e = datetime.strptime(flist[i].split("_e")[1].split("_c")[0][:-1],time_format)
                    scan_start[counter_idx] = scan_s
                    scan_end[counter_idx] = scan_e
                    datestr[counter_idx] = datetime.strftime(scan_s,"%Y%m%d")
                    time_ind[counter_idx] = i
                    counter_idx =+ 1

    # compute central scantime
    time = scan_start + (scan_end - scan_start)/2

    # generate Dataset
    goes_ref_ds = xr.Dataset(data_vars=dict(starttime_scan = (["time"], scan_start),
                               endtime_scan = (["time"], scan_end),
                               datestring = (["time"], datestr),
                               t_index = (["time"], time_ind)),
                            coords=dict(
                                time = time
                            ),
                attrs=dict(description="GOES image times and image file reference"),
                )
    
    goes_ref_ds["time"] = goes_ref_ds.time.assign_attrs(description="central time of scan (time of tropics scan)")

    return goes_ref_ds.sortby("time")






