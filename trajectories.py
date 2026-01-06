"""
-------------------------------------------------------------------------------------------
Trajectory preprocessing functions for filtering and interpolating ERA5 trajectories.
-------------------------------------------------------------------------------------------
"""
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline
from datetime import datetime,timedelta

from .utils import where_both, dropna, generate_globsearch_string, generate_url_list

def main():
    # input directory
    traj_dir = "~/PhD/Datasets/orig_trajectories/"
    ref_dir = "~/Data/goes16_reference/"
    traj_file_name = "NAtl_Trajectories_Mid_Start_925hPa_1hrLocalInterp_ERA5_vars_Dec-Feb_2017-2022"
    generate_ref_ds = True
    add_dt = True
    int_skt_sst = False     # Interpolation of skin temperature and sst
    data_source = "ERA5" # "GOES"

    # get datasets
    trajects = xr.open_dataset(traj_dir + traj_file_name + ".nc")

    if generate_ref_ds or data_source != "GOES":
        print(str(datetime.now())+": Generating reference dataset...")
        if data_source == "GOES":
            goes_ref_ds = get_goes_ref_ds(["2017","2018","2019","2020","2021","2022"],[12,1,2])
            goes_ref_ds.to_netcdf(ref_dir + "goes_ref_ds.nc")
            data_ref_time = goes_ref_ds.time.values
        elif data_source == "ERA5":
            data_ref_time = get_era_ref_times(np.arange(2016,2023))
    else:
        goes_ref_ds = xr.open_dataset(ref_dir + "goes_ref_ds.nc")
        data_ref_time = goes_ref_ds.time.values

    if add_dt:
        print(str(datetime.now())+": Adding UTC datetime to trajectories...")
        trajects = add_datetime(trajects)

    print(str(datetime.now())+": Interpolate trajectories...")
    trajects = interpolate_trajects(trajects,data_ref_time,int_skt_sst)
    trajects.to_netcdf(traj_dir + traj_file_name + "_intp.nc")


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
    
    # get dimension names
    dims = list(trajects.dims)[:2]

    time_dim = next((d for d in dims if "time" in d.lower()), None)
    if time_dim is None:
        raise NameError("Time dimension not found.")
    
    traj_dim = next(d for d in dims if d!= time_dim)

    # print for debugging
    print(f"time_dim={time_dim}, traj_dim={traj_dim}")

    # get number of trajectories / days    
    N_trajects = trajects.sizes[traj_dim]
    
    for i in range(N_trajects):
        traject = trajects.isel({traj_dim: i})
        
        # get lat lon tendencies
        lat = traject.latitude.dropna(dim=time_dim,how="all").values
        lon = traject.longitude.dropna(dim=time_dim,how="all").values
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
    trajects_loop_free = trajects.isel({traj_dim: no_loops_idx})
    trajects_loop_cw = trajects.isel({traj_dim: cw_loops_idx})
    trajects_loop_ccw = trajects.isel({traj_dim: ccw_loops_idx})

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

def interpolate_trajects(trajects,data_ref_time,int_skt_sst=False,N_timesteps=960):
    """
    Interpolates trajectories linearly from 1 hourly trajectories onto the 10/15-min GOES images
    or other given data reference times.
    """
    
    # Get trajectory numbers and amount of them
    Trajectory_N = trajects.Trajectory_N.values
    N_Trajectories = len(Trajectory_N)
    
    # Initialize arrays to build new interpolated Dataset
    longitudes = np.full((N_timesteps,N_Trajectories),np.nan)
    latitudes = np.full((N_timesteps,N_Trajectories),np.nan)
    datetime_UTC = np.full((N_timesteps,N_Trajectories),np.nan).astype("datetime64[ns]")
    sst = np.full((N_timesteps,N_Trajectories),np.nan)
    skt = np.full((N_timesteps,N_Trajectories),np.nan)
    
    for i in range(N_Trajectories):
        # select trajectory 
        traject = trajects.isel(N_Trajectories=i)#.dropna(dim="Hours_Local_Time")   # Note: by dropping NaNs here we lose alignment by local hour
        
        # get trajectory times and locations
        traj_times = traject.datetime_UTC.values
        traj_lons = traject.longitude.values
        traj_lats = traject.latitude.values

        if int_skt_sst:
            # get trajectory sst
            traj_sst = traject.SST.values
            traj_sst_times = traject.datetime_UTC.values[np.isfinite(traj_sst)]
            traj_sst = traj_sst[np.isfinite(traj_sst)]
            
            # get trajectory skt
            traj_skt = traject.skt.values
            traj_skt_times = traject.datetime_UTC.values[np.isfinite(traj_skt)]
            traj_skt = traj_skt[np.isfinite(traj_skt)]

        # interpolate trajectory onto GOES scantimes
        traj_interp = temp_interp_2D(traj_times,traj_lons,traj_lats,data_ref_time)

        datetime_UTC[:len(traj_interp[0]),i] = traj_interp[0]
        longitudes[:len(traj_interp[1]),i] = traj_interp[1]
        latitudes[:len(traj_interp[2]),i] = traj_interp[2]

        if int_skt_sst:
            # interpolate sst onto GOES scantimes (cubic spline interpolation)
            t_interp = data_ref_time[(data_ref_time>traj_sst_times[0])*(data_ref_time<traj_sst_times[-1])]
            sst[:len(t_interp),i] = spline_interp(traj_sst_times,traj_sst,t_interp)
            t_interp = data_ref_time[(data_ref_time>traj_skt_times[0])*(data_ref_time<traj_skt_times[-1])]
            skt[:len(t_interp),i] = spline_interp(traj_skt_times,traj_skt,t_interp)

    if int_skt_sst:
        ds = xr.Dataset(data_vars=dict(Trajectory_N=(["N_Trajectories"],Trajectory_N),
                                        longitude=(["Time","N_Trajectories"],longitudes),
                                        latitude=(["Time","N_Trajectories"],latitudes),
                                        datetime_UTC=(["Time","N_Trajectories"],datetime_UTC),
                                        sst=(["Time","N_Trajectories"],sst),
                                        skt=(["Time","N_Trajectories"],skt)),
                        attrs=dict(description="Trajectory data interpolated on GOES images"))
    else: 
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

def spline_interp(t_data,data,t_interp):
    """
    Spline interpolation using CubicSpline from Scipy. 

    Input:
    -----------------------------------------------------------------------------
    - t_data                independent variable (e.g. times with available data)
    - data                  dependent variable
    - t_interp              times to interpolate on
    
    Output:
    -----------------------------------------------------------------------------
    - interp_data           interpolated data
    """
    cs = CubicSpline(t_data,data)
    return cs(t_interp)

def get_era_ref_times(years):
    """
    Generate array of times with ERA5 data from given years.
    Input:
    ---------------------------------------------------------------------------------------
    - years:            List of int giving the years of interest

    Output:
    ---------------------------------------------------------------------------------------
    - era_ref_times     Array of UTC times of ERA 5 data
    """
    if isinstance(years, int):  # allow single int
        years = [years]

    hours = []

    for y in years:
        # append hours
        hours.append(np.arange(f"{y}-01-01", f"{y}-03-05", dtype="datetime64[h]"))
        hours.append(np.arange(f"{y}-11-27", f"{y+1}-01-01", dtype="datetime64[h]"))

    return np.sort(np.concatenate(hours)).astype("datetime64[ns]")


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
    scan_start = np.full(max_length, np.nan, dtype="datetime64[ns]") 
    scan_end = np.full(max_length, np.nan, dtype="datetime64[ns]") 
    datestr = np.full(max_length, np.nan) 
    time_ind = np.full(max_length, np.nan) 
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
                    counter_idx += 1

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
                    counter_idx += 1
    
    # exclude nans and convert dtype
    scan_start = dropna(scan_start)
    scan_end = dropna(scan_end)
    datestr = dropna(datestr).astype(int).astype(str)
    time_ind = dropna(time_ind).astype(int)

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


if __name__ == "__main__":
    main()






