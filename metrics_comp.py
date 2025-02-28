"""
Computation of cloudmetrics along interpolated ERA5 trajectories.
"""

import numpy as np
import xarray as xr
import cloudmetrics
import time 
from datetime import datetime
import glob
import os
import resource
#from line_profiler import LineProfiler

#from gogoesgone import processing as pr
#from gogoesgone import zarr_access as za

from utils import where_both, extract_frame
from trajectories import interpolate_trajects, add_datetime

def main():
    start = time.time()
    # initialize line profiler
    # lp = LineProfiler()
    # lp_wrapper = lp(compute_metrics)

    # set inputs
    data_dir = "/scratch-shared/rmeier/Data/GOES-CMIP-C13-Tropical-North-Atlantic/daily/"
    traj_dir = "data/trajectories/"
    traj_file_name = "NAtl_Trajectories_Mid_Start_925hPa_1hrLocalInterp_ERA5_vars_Dec-Feb_2020"
    save_freq = 100 # save after every 100 images
    framesize = 5

    # get datasets
    goes_ref_ds = xr.open_dataset("data/goes_reference/goes_ref_ds.nc")
    trajects = xr.open_dataset(traj_dir + traj_file_name + ".nc")

    print(str(datetime.now())+": Adding UTC datetime to trajectories...")
    trajects = add_datetime(trajects)

    print(str(datetime.now())+": Interpolate trajectories...")
    trajects = interpolate_trajects(trajects,goes_ref_ds)
    
    print(str(datetime.now())+": Selecting trajectory extent...")
    traj_extents = (-70+framesize/2,-10-framesize/2,0+framesize/2,40-framesize/2) 
    trajects = trajects.where((trajects.longitude >= traj_extents[0])
                             &(trajects.longitude <= traj_extents[1])
                             &(trajects.latitude >= traj_extents[2])
                             &(trajects.latitude <= traj_extents[3]))  
    
    # compute
    res_trajects = compute_metrics(trajects,goes_ref_ds,data_dir,traj_dir,framesize,save_freq,accessmode="no access")
    #res_trajects = lp_wrapper(trajects,goes_ref_ds,framesize,save_freq)
    os.remove(traj_dir + traj_file_name + "_with_metrics.nc")
    res_trajects.to_netcdf(traj_dir + traj_file_name + "_with_metrics.nc")
    #lp.print_stats()
    print("programm completed in" + str(round(time.time()-start,0)) + "s.")
    print("memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "Kb")


def compute_metrics(trajects,goes_ref_ds,data_dir,traj_dir,traj_file_name,framesize=5,save_freq=100,accessmode="netCDF"):
    """
    Computes metrics on interpolated trajectories and gives out the trajectory dataset with metrics.

    Parameters
    ----------
    trajects: xarray.Dataset with dimensions "N_Trajectories" and "Time"
        Interpolated trajectories with latitude, longitude and UTC time along the "Time" dimension
    goes_ref_ds: xarray.Dataset with dimension "time"
        Dataset of GOES-16 images of winter seasons (DJF) between 2017/18 and 2022/23 with start-, end- and centraltime of the scan and datestring and t_index to allocate the image in downloaded files
    framesize: int
        Size of the square lat/lon frame in degree, where the metrics are computed on (standard 5x5)
    accessmode: "netCDF" or "json"
        Mode of access via downloaded netCDF files or via json mapping to AWS server, where GOES-16 data is stored
        (Currently json mode is not implemented!)

    Return
    ------
    trajects: xarray.Dataset with dimensions "N_Trajectories" and "Time"
        Same dataset as input, but with metrics added as variables along the "Time" dimension
    """
    print(str(datetime.now())+": Initialize cloud metrics...")

    # Get number and maximum size of the trajectories
    N_Trajectories = trajects.sizes["N_Trajectories"]
    N_timesteps = trajects.sizes["Time"]
    
    # initialize metric arrays 
    open_sky = np.full((N_timesteps,N_Trajectories),np.nan)
    L_max = np.full((N_timesteps,N_Trajectories),np.nan)
    L_mean = np.full((N_timesteps,N_Trajectories),np.nan)
    cloud_frac = np.full((N_timesteps,N_Trajectories),np.nan)
    num_objects = np.full((N_timesteps,N_Trajectories),np.nan)
    orientation = np.full((N_timesteps,N_Trajectories),np.nan)
    fractal_dim = np.full((N_timesteps,N_Trajectories),np.nan)
    mean_perimeter = np.full((N_timesteps,N_Trajectories),np.nan)
    cop = np.full((N_timesteps,N_Trajectories),np.nan)
    iorg = np.full((N_timesteps,N_Trajectories),np.nan)
    scai = np.full((N_timesteps,N_Trajectories),np.nan)
    hcf_280 = np.full((N_timesteps,N_Trajectories),np.nan)
    hcf_282 = np.full((N_timesteps,N_Trajectories),np.nan)
    hcf_285 = np.full((N_timesteps,N_Trajectories),np.nan)
    
    # get the relevant GOES image period
    goes_ref_ds = goes_ref_ds.sel(time=slice(trajects.datetime_UTC.min(skipna=True).values,
                                             trajects.datetime_UTC.max(skipna=True).values))
    
    print(str(datetime.now())+": Initialization done")

    # iterating through the relevant GOES images
    for goes_index, goes_time in enumerate(goes_ref_ds.time):
         
        # checking which trajectories are on this image
        time_indices, traj_indices = np.where(trajects.datetime_UTC == goes_time)

        # exclude images where no trajectories are found
        if len(traj_indices) == 0:
            continue
        else:
            datestring = str(goes_ref_ds.sel(time=goes_time).datestring.values)
            t_index = goes_ref_ds.sel(time=goes_time).t_index.values

            if accessmode == "netCDF":
                dayoftheyearstring = datetime.strptime(datestring,"%Y%m%d").strftime("%j")
                yearstring = datetime.strptime(datestring,"%Y%m%d").strftime("%Y")
                file_name = data_dir + yearstring + "/OR_ABI-L2-CMIPF-M6C13_G16_"  + dayoftheyearstring +".nc"
                #file_name = "/home/robert/Coding/cloud_org_evolution/goes16_CMI/CMI_subset_" + datestring + "_goes16_ABI-L2-CMIPF_13.nc"

                if len(glob.glob(file_name)) == 0:
                    continue

                elif goes_index == 0:    # first image 
                    CMIPF_day = xr.open_dataset(file_name).sortby("t")
                    datestring_old = datestring

                #TODO: Check if this elif statement can be removed
                elif datestring != goes_ref_ds.isel(time=goes_index-1).datestring.values:  # changing datestring
                    CMIPF_day = xr.open_dataset(file_name).sortby("t")

                elif datestring != datestring_old: # changed datestring (also after skipping images)
                    CMIPF_day = xr.open_dataset(file_name).sortby("t")
            
                # updating old datestring
                datestring_old = datestring
                
                # selecting image
                CMIPF = CMIPF_day.sel(t=goes_time,method="nearest")

            elif accessmode == "no access":
                continue
            
            else:
                raise TypeError("Currently, only netCDF is implemented as accessmode.")

            # iterating over active trajectories
            for i in range(len(traj_indices)):
                traj = trajects.isel(dict(Time=time_indices[i],N_Trajectories=traj_indices[i]))
                
                # cut out subset
                extent = np.array([traj.longitude - framesize/2,
                                   traj.longitude + framesize/2,
                                   traj.latitude - framesize/2, 
                                   traj.latitude + framesize/2])
                if accessmode == "netCDF":
                    CMIP = extract_frame(CMIPF,extent)
                
                # can be removed if not used anymore
                elif accessmode == "json":
                    CMIP = CMIPF.subset_region_from_latlon_extents(extent, unit="degree")
                
                # compute cloud mask
                CMI = CMIP.CMI.values
                mask = np.where(np.isfinite(CMI),1*((CMI < 290)&(CMI > 280)),CMI)
                
                # compute the metrics
                cloud_frac[time_indices[i],traj_indices[i]] = cloudmetrics.mask.cloud_fraction(mask=mask)
                num_objects[time_indices[i],traj_indices[i]] = cloudmetrics.mask.num_objects(mask=mask, periodic_domain=False)
                fractal_dim[time_indices[i],traj_indices[i]] = cloudmetrics.mask.fractal_dimension(mask=mask)
                hcf_280[time_indices[i],traj_indices[i]] = high_cloud_fraction(CMI,280)
                hcf_282[time_indices[i],traj_indices[i]] = high_cloud_fraction(CMI,282)
                hcf_285[time_indices[i],traj_indices[i]] = high_cloud_fraction(CMI,285)
                
                # compute object-based metrics if there are objects
                if len(np.where(mask==1)[0]) > 0:
                    L_max[time_indices[i],traj_indices[i]] = cloudmetrics.mask.max_object_length_scale(mask=mask, periodic_domain=False)
                    L_mean[time_indices[i],traj_indices[i]] = cloudmetrics.mask.mean_object_length_scale(mask=mask, periodic_domain=False)
                    #orientation[time_indices[i],traj_indices[i]] = cloudmetrics.mask.orientation(mask=mask, periodic_domain=False)
                    mean_perimeter[time_indices[i],traj_indices[i]] = cloudmetrics.mask.mean_object_perimeter_length(mask=mask, periodic_domain=False)
                    cop[time_indices[i],traj_indices[i]] = cloudmetrics.mask.cop_objects(mask=mask, periodic_domain=False)
                    iorg[time_indices[i],traj_indices[i]] = cloudmetrics.mask.iorg_objects(mask=mask, periodic_domain=False)
                    scai[time_indices[i],traj_indices[i]] = cloudmetrics.mask.scai_objects(mask=mask, periodic_domain=False)
                
                # compute open sky with another mask
                mask = np.where(np.isfinite(CMI),1*((CMI < 290)&(CMI > 280)),1)
                open_sky[time_indices[i],traj_indices[i]] = cloudmetrics.mask.open_sky(mask=mask, periodic_domain=False)
        
        if (goes_index%save_freq) == 0:
            print(str(datetime.now())+": Saving...")
            res_trajects = trajects.assign(variables=dict(L_max=(["Time","N_Trajectories"],L_max),
                                              L_mean=(["Time","N_Trajectories"],L_mean),
                                              Cloud_fraction=(["Time","N_Trajectories"],cloud_frac),
                                              Number_of_objects=(["Time","N_Trajectories"],num_objects),
                                              Orientation=(["Time","N_Trajectories"],orientation),
                                              Fractal_dimension=(["Time","N_Trajectories"],fractal_dim),
                                              Mean_perimeter=(["Time","N_Trajectories"],mean_perimeter),
                                              COP=(["Time","N_Trajectories"],cop),
                                              Iorg=(["Time","N_Trajectories"],iorg),
                                              SCAI=(["Time","N_Trajectories"],scai),
                                              HCF_280K=(["Time","N_Trajectories"],hcf_280),
                                              HCF_282K=(["Time","N_Trajectories"],hcf_282),
                                              HCF_285K=(["Time","N_Trajectories"],hcf_285),
                                              Open_sky=(["Time","N_Trajectories"],open_sky)))
            if len(glob.glob(traj_dir + traj_file_name + "_with_metrics.nc")) > 0:
                os.remove(traj_dir + traj_file_name +"_with_metrics.nc")
            res_trajects.to_netcdf(traj_dir + traj_file_name +"_with_metrics.nc")

        print(str(datetime.now())+": " +str(int((goes_index+1)/len(goes_ref_ds.datestring)*100)) + "% done",end="\r")
        
    res_trajects = trajects.assign(variables=dict(L_max=(["Time","N_Trajectories"],L_max),
                                              L_mean=(["Time","N_Trajectories"],L_mean),
                                              Cloud_fraction=(["Time","N_Trajectories"],cloud_frac),
                                              Number_of_objects=(["Time","N_Trajectories"],num_objects),
                                              Orientation=(["Time","N_Trajectories"],orientation),
                                              Fractal_dimension=(["Time","N_Trajectories"],fractal_dim),
                                              Mean_perimeter=(["Time","N_Trajectories"],mean_perimeter),
                                              COP=(["Time","N_Trajectories"],cop),
                                              Iorg=(["Time","N_Trajectories"],iorg),
                                              SCAI=(["Time","N_Trajectories"],scai),
                                              HCF_280K=(["Time","N_Trajectories"],hcf_280),
                                              HCF_282K=(["Time","N_Trajectories"],hcf_282),
                                              HCF_285K=(["Time","N_Trajectories"],hcf_285),
                                              Open_sky=(["Time","N_Trajectories"],open_sky)))    
        
        
    return res_trajects


def high_cloud_fraction(temperatures,threshold):
    """
    Computes the fraction of pixels with lower brightness temperature 
    (therefore higher in altitude) than a threshold temperature.
    
    Parameters
    ----------
    temperatures: array_like
        Brightness temperatures per pixel from satellite image
    threshold: scalar
        Threshold temperature (exclusive)

    Return
    ------
    fraction of pixels below threshold temperature
    """
    return np.histogram(temperatures, bins=[0,threshold,400],density=True)[0][0]*threshold


if __name__ == "__main__":
    main()
