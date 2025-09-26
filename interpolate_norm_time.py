import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astral import LocationInfo
import datetime
from astral.sun import sun
from functools import lru_cache
import resource
import time 

def main():
    has_norm_time = True

    if has_norm_time == False:
        # load cloudmetric dataset
        file_path = "/home/rmeier1/PhD/Datasets/Catalogue_upload/Lagrangian_cloudmetrics_rounded_N.nc"
        cloudmetric_ds = xr.open_dataset(file_path)

        start = time.time()

        # add normalized time to dataset 
        print("Adding normalized time...")
        cloudmetric_ds = add_norm_time(cloudmetric_ds)

        # save intermediate step
        save_path = "/home/rmeier1/PhD/Datasets/interp_data/cloudmetrics_with_norm_time.nc"
        cloudmetric_ds.to_netcdf(save_path)

    else:
        file_path = "/home/rmeier/Data/Metrics/cloudmetrics_with_norm_time.nc"
        cloudmetric_ds = xr.open_dataset(file_path)


    # interpolate dataset on normalized time
    print("Interpolating...")
    cloudmetric_ds_interp = interpolate_dataset(cloudmetric_ds,interp_method="numpy")

    # saving dataset 
    save_path = "/home/rmeier/Data/Metrics/diurnal_cloudmetrics.nc"
    cloudmetric_ds_interp.to_netcdf(save_path)

    print("programm completed in" + str(round(time.time()-start,0)) + "s.")
    print("memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "Kb")

def add_norm_time(ds):
    """
    Assign normalized time, day ID and local date of sunset to dataset.
    """
    # get dimension names
    time_dim, traj_dim = list(ds.dims)[:2]
    if not time_dim in ["Time","Hours_Local_Time"]:
        if traj_dim in ["Time","Hours_Local_Time"]:
            traj_dim, time_dim = list(ds.dims)[:2]
        else: 
            raise NameError("Time dimension not one of [Hours_Local_Time, Time].")

    # get coordinates, UTC datetimes and trajectory number
    lon = ds.longitude.values
    lat = ds.latitude.values
    datetime_UTC = ds.datetime_UTC.values
    Trajectory_N = ds.N_Trajectories.values

    # compute norm. time and local date
    norm_time, local_date = convert_UTC_to_norm_time(lon,lat,datetime_UTC)

    # get day of trajectory
    day_of_traj = (local_date - np.nanmin(local_date,axis=0))

    # assemble day_id
    day_id = np.where(np.isfinite(day_of_traj),Trajectory_N.astype(str) + "D" + day_of_traj.astype(int).astype(str),"NaNDNaN")

    # assign coordinates
    ds = ds.assign_coords({"norm_time":(list(ds.dims)[:2], norm_time), 
                           "day_id":(list(ds.dims)[:2],day_id), 
                           "local_date":(list(ds.dims)[:2],local_date)})

    return ds


def interpolate_dataset(ds,interp_method,t_res=0.01):
    """
    Compute interpolated dataset variables on normalized times of given resolution.
    """
    # get sizes of interpolated dataset
    day_id_unique = np.unique(ds.day_id.values.T)[:-1]
    N_days = len(day_id_unique)

    t_interp_day = np.arange(0,1,t_res)
    N_norm_time = len(t_interp_day)

    # initialize dictionary of coordinates
    coords_dict = {}
    coords_dict["longitude"] = (["Norm_Time","Day_ID"],np.full(shape=(N_norm_time,N_days),fill_value=np.nan))
    coords_dict["latitude"] = (["Norm_Time","Day_ID"],np.full(shape=(N_norm_time,N_days),fill_value=np.nan))
    coords_dict["Norm_Time"] = (["Norm_Time"],t_interp_day)
    coords_dict["Day_ID"] = (["Day_ID"],day_id_unique)
    coords_dict["local_date"] = (["Day_ID"],np.full(shape=(N_days),fill_value=np.nan,dtype="datetime64[ns]"))
    coords_dict["Mask"] = (["Mask"],ds.Mask.values)

    # initialize dictionary of interpolated variables
    var_dict = {}
    for var_name in list(ds):
        if len(ds[var_name].shape) > 2:     # mask dependent variables
            var_dict[var_name] = (["Norm_Time","Day_ID","Mask"],np.full(shape=(N_norm_time,N_days,2),fill_value=np.nan))
        else:                               # mask independent variables
            var_dict[var_name] = (["Norm_Time","Day_ID"],np.full(shape=(N_norm_time,N_days),fill_value=np.nan))

    # iterate through trajectories
    for n in range(ds.sizes["N_Trajectories"]):
        ts = ds.isel(N_Trajectories=n).dropna(dim="Time",how="all")

        # get 1D time coordinates
        norm_time = ts.norm_time.values
        day_id = ts.day_id.values
        day_of_traj = np.char.partition(day_id,"D")[:,2].astype(float)

        # compute normalized time of trajectory (monotone increase) and interpolation range
        traj_norm_time = norm_time + day_of_traj
        nan_stop = np.argwhere(np.diff(traj_norm_time)>0.1)
        if len(nan_stop>0):
            nan_stop = traj_norm_time[nan_stop[0][0]]+t_res
            t_interp = np.arange(np.ceil(np.nanmin(traj_norm_time)*100)/100,nan_stop,t_res)
        else:
            t_interp = np.arange(np.ceil(np.nanmin(traj_norm_time)*100)/100,np.nanmax(traj_norm_time),t_res)

        # compute indices for saving
        day_idx = np.searchsorted(np.unique(day_of_traj),t_interp.astype(int))
        norm_time_idx = ((t_interp%1)*100).astype(int)
        day_id_indices = np.searchsorted(day_id_unique,np.unique(day_id)[day_idx])
        local_day_indices = np.searchsorted(day_id,np.unique(day_id)[day_idx])

        # save local date coordinate
        coords_dict["local_date"][1][day_id_indices] = ts.local_date.values[local_day_indices]

        if interp_method == "numpy":
            # interpolate metrics and lat/lon
            for coord_name in ["longitude","latitude"]:
                coord_intp = np.interp(t_interp,traj_norm_time,ts[coord_name])
                coords_dict[coord_name][1][norm_time_idx,day_id_indices] = coord_intp

            # interpolate variables
            for var_name in list(ds):
                if len(ds[var_name].shape) > 2:     # mask dependent variables
                    var_dict[var_name][1][norm_time_idx,day_id_indices,0] = np.interp(t_interp,traj_norm_time,ts[var_name].isel(Mask=0))
                    var_dict[var_name][1][norm_time_idx,day_id_indices,1] = np.interp(t_interp,traj_norm_time,ts[var_name].isel(Mask=1))
                else:                               # mask independent variables
                    var_dict[var_name][1][norm_time_idx,day_id_indices] = np.interp(t_interp,traj_norm_time,ts[var_name])
        
        elif interp_method == "xarray":
            # traj_norm_time to ts
            ts = ts.assign_coords({"Time":(["Time"],traj_norm_time)})

            # interpolate timeseries
            ts_interp = ts.interp(Time=t_interp)
            
            # assign to dictionaries
            for coord_name in ["longitude","latitude"]:
                coords_dict[coord_name][1][norm_time_idx,day_id_indices] = ts_interp[coord_name].values
            for var_name in list(ds):
                var_dict[var_name][1][norm_time_idx,day_id_indices] = ts_interp[var_name].values

        else:
            raise ValueError("Give interpolation method numpy or xarray.")

        print(f"Interpolation: {np.round(n/ds.sizes['N_Trajectories']*100,1)}% complete",end="\r")

    ds_interp = xr.Dataset(coords=coords_dict,
                           data_vars=var_dict)

    return ds_interp


def convert_UTC_to_norm_time(lon,lat,datetime_UTC):
    """
    Convert UTC time of a trajectory point into a normalized time relative to sunset and sunrise.
    Additionally return local date of sunrise
    """
    # Round coordinates to avoid unnecessary recomputations
    lon = np.round(lon,1)
    lat = np.round(lat,1)

    sunset, sunrise, next_sunset = find_sunset_sunrise_bounds(lon,lat,datetime_UTC)

    # compute local date of sunrise
    local_date = (sunrise + lon/180*np.timedelta64(12*60*60,"s")).astype("datetime64[D]")

    # compute time relative to sunset and sunrise
    time_since_sunset = datetime_UTC - sunset 
    sunset_to_sunrise = sunrise-sunset
    sunset_to_sunset = next_sunset-sunset

    norm_time_night = time_since_sunset/sunset_to_sunrise * 0.5
    norm_time_day = 0.5 + (time_since_sunset-sunset_to_sunrise)/(sunset_to_sunset-sunset_to_sunrise) * 0.5
    norm_time = np.where(time_since_sunset <= sunset_to_sunrise,norm_time_night,norm_time_day)

    return norm_time, local_date


def find_sunset_sunrise_bounds(lon,lat,datetime_UTC):
    """ 
    Iterate over array to find previous sunsets and subsequent sunrise and sunset times.
    """
    # get UTC date and date before and after
    date_UTC = datetime_UTC.astype("datetime64[D]")
    date_before = date_UTC - np.timedelta64(1,"D")
    date_after = date_UTC + np.timedelta64(1,"D")

    # Get array shape
    N_t, N_traj = np.shape(date_UTC)

    # Initialize output
    prev_sunset = np.full(shape=(N_t,N_traj),fill_value=np.nan,dtype="datetime64[ns]")
    subs_sunrise = np.full(shape=(N_t,N_traj),fill_value=np.nan,dtype="datetime64[ns]")
    next_sunset = np.full(shape=(N_t,N_traj),fill_value=np.nan,dtype="datetime64[ns]")

    for i in range(N_traj):
        for j in range(N_t):
            # skip nan values
            if np.isnan(lon[j,i]):
                continue

            # compute previous sunset and subsequent sunrise in UTC
            sunrise, sunset = sunrise_sunset_UTC_time(lon[j,i],lat[j,i],date_UTC[j,i])

            if sunset > sunrise:            # sunset not past UTC date boundary yet
                if datetime_UTC[j,i] < sunset:       # time before sunset
                    sunset = sunrise_sunset_UTC_time(lon[j,i],lat[j,i],date_before[j,i])[1]
                else:                           # time past sunset
                    sunrise = sunrise_sunset_UTC_time(lon[j,i],lat[j,i],date_after[j,i])[0]

            else:                           # sunset past UTC date boundary
                if datetime_UTC[j,i] < sunset:       # time before sunset
                    sunrise, sunset = sunrise_sunset_UTC_time(lon[j,i],lat[j,i],date_before[j,i])

            next_sunset[j,i] = sunrise_sunset_UTC_time(lon[j,i],lat[j,i],sunset.astype("datetime64[D]") + np.timedelta64(1,"D"))[1]
            prev_sunset[j,i] = sunset
            subs_sunrise[j,i] = sunrise

        print(f"Chasing sunsets and sunrises: {np.round((i+1)/N_traj*100,1)}% complete",end='\r')

    return prev_sunset, subs_sunrise, next_sunset

@lru_cache(maxsize=None)
def sunrise_sunset_UTC_time(lon,lat,date_UTC):
    """
    Use astral package to compute local sunrise/sunset UTC times on UTC date.
    """
    # convert from numpy.datetime64 to datetime.date
    date_UTC = date_UTC.astype(datetime.date)

    # find UTC times of local sunrise and sunset and convert back to numpy.datetime64
    loc = LocationInfo(latitude=lat, longitude=lon)
    s = sun(loc.observer, date=date_UTC, tzinfo=loc.timezone)

    sunrise_UTC = np.datetime64(s['sunrise'].replace(tzinfo=None))
    sunset_UTC = np.datetime64(s['sunset'].replace(tzinfo=None)) 

    return sunrise_UTC, sunset_UTC



if __name__ == "__main__":
    main()
