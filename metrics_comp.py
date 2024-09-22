import numpy as np
import xarray as xr
import cloudmetrics
import time 
import datetime
import glob
import os
import resource
from line_profiler import LineProfiler

from gogoesgone import processing as pr
from gogoesgone import zarr_access as za

def where_both(condition_1,condition_2):
    return np.where(np.where(condition_1,True,False)*np.where(condition_2,True,False))

def high_cloud_fraction(temperatures,threshold):
    """
    Computes the fraction of pixels with lower brightness temperature (therefore higher in altitude) than a threshold temperature.
    
    Parameters
    ----------
    temperatures: array_like
        Brightness temperatures per pixel in a satellite image
    threshold: scalar
        Threshold temperature 

    Return
    ------
    fraction of pixels below threshold temperature
    """
    return np.histogram(temperatures, bins=[0,threshold,400],density=True)[0][0]*threshold

def compute_metrics(trajects,goes_ref_ds,framesize=5,save_freq=100,accessmode="netCDF"):
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

    Return
    ------
    trajects: xarray.Dataset with dimensions "N_Trajectories" and "Time"
        Same dataset as input, but with metrics added as variables along the "Time" dimension
    """
    print(str(datetime.datetime.now())+": Initialize...")
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
    t_img_i = trajects.datetime_UTC.min(skipna=True)
    t_img_f = trajects.datetime_UTC.max(skipna=True)
    goes_ref_ds = goes_ref_ds.isel(time=where_both(goes_ref_ds.middletime_scan >= t_img_i, goes_ref_ds.middletime_scan <= t_img_f)[0])
    
    print(str(datetime.datetime.now())+": Initialization done")
    # iterating through the relevant GOES images
    for goes_index in range(goes_ref_ds.sizes["time"]):
         
        # checking which trajectories are on this image
        time_indices, traj_indices = np.where(trajects.datetime_UTC == goes_ref_ds.middletime_scan.isel(time=goes_index).values)

        if len(traj_indices) == 0:
            continue
        else:
            datestring = str(goes_ref_ds.isel(time=goes_index).datestring.values)
            t_index = goes_ref_ds.isel(time=goes_index).t_index.values

            if accessmode == "netCDF":
                #dayoftheyearstring = datetime.datetime.strptime(datestring,"%Y%m%d").strftime("%j")
                #yearstring = datetime.datetime.strptime(datestring,"%Y%m%d").strftime("%Y")
                #file_name = "/net/labdata/geet/Data/GOES-CMIP-C13-Tropical-North-Atlantic/daily/"+yearstring+"/OR_ABI-L2-CMIPF-M6C13_G16_"  + dayoftheyearstring +".nc"
                file_name = "/home/robert/Coding/cloud_org_evolution/goes16_CMI/CMI_subset_" + datestring + "_goes16_ABI-L2-CMIPF_13.nc"

                if len(glob.glob(file_name)) == 0:
                    continue

                elif goes_index == 0:    # first image 
                    img_day = xr.open_dataset(file_name).sortby("t")
                    datestring_old = datestring

                elif datestring != goes_ref_ds.isel(time=goes_index-1).datestring.values:  # changing datestring
                    img_day = xr.open_dataset(file_name).sortby("t")

                elif datestring != datestring_old: # changed datestring after skipping images
                    img_day = xr.open_dataset(file_name).sortby("t")
            
                # updating old datestring
                datestring_old = datestring
                
                # selecting image
                img = img_day.isel(t=t_index)

            else:
                print("Use json or netCDF as accessmode.")

            # iterating over active trajectories
            for i in range(len(traj_indices)):
                track = trajects.isel(dict(Time=time_indices[i],N_Trajectories=traj_indices[i]))
                
                # cut out subset
                extent = np.array([track.longitude-framesize/2,track.longitude+framesize/2,track.latitude-framesize/2,track.latitude+framesize/2])
                if accessmode == "json":
                    subset = img.subset_region_from_latlon_extents(extent, unit="degree")
                elif accessmode == "netCDF":
                    subset = pr.subset_region_from_latlon_extents(img,extent, unit="degree")
                
                # compute cloud mask
                subset_array = subset.CMI.values
                mask = np.where(np.isfinite(subset_array),1*((subset_array < 290)&(subset_array > 280)),subset_array)
                
                # compute the metrics
                cloud_frac[time_indices[i],traj_indices[i]] = cloudmetrics.mask.cloud_fraction(mask=mask)
                num_objects[time_indices[i],traj_indices[i]] = cloudmetrics.mask.num_objects(mask=mask, periodic_domain=False)
                fractal_dim[time_indices[i],traj_indices[i]] = cloudmetrics.mask.fractal_dimension(mask=mask)
                hcf_280[time_indices[i],traj_indices[i]] = high_cloud_fraction(subset_array,280)
                hcf_282[time_indices[i],traj_indices[i]] = high_cloud_fraction(subset_array,282)
                hcf_285[time_indices[i],traj_indices[i]] = high_cloud_fraction(subset_array,285)
                
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
                mask = np.where(np.isfinite(subset_array),1*((subset_array < 290)&(subset_array > 280)),1)
                open_sky[time_indices[i],traj_indices[i]] = cloudmetrics.mask.open_sky(mask=mask, periodic_domain=False)
        
        if (goes_index%save_freq) == 0:
            print(str(datetime.datetime.now())+": Saving...")
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
            if len(glob.glob("./trajectories_with_metrics.nc")) > 0:
                os.remove("trajectories_with_metrics.nc")
            res_trajects.to_netcdf("trajectories_with_metrics.nc")

        print(str(datetime.datetime.now())+": " +str(int((goes_index+1)/len(goes_ref_ds.datestring)*100)) + "% done")
        
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

if __name__ == "__main__":

    start = time.time()
    # initialize line profiler
    lp = LineProfiler()
    lp_wrapper = lp(compute_metrics)
    # get datasets and set input 
    goes_ref_ds = xr.open_dataset("goes_ref_ds.nc")
    trajects = xr.open_dataset("trajectory.nc")
    save_freq = 100 # save after every 100 images
    framesize = 5
    traj_extents = (-60+framesize/2,-20-framesize/2,-5+framesize/2,30-framesize/2) 
    trajects = trajects.where((trajects.longitude >= traj_extents[0])
                             &(trajects.longitude <= traj_extents[1])
                             &(trajects.latitude >= traj_extents[2])
                             &(trajects.latitude <= traj_extents[3]),
                             drop=True)  # Should not be dropped for alignment
    # compute
    #res_trajects =  compute_metrics(trajects,goes_ref_ds,framesize,save_freq)
    res_trajects =  lp_wrapper(trajects,goes_ref_ds,framesize,save_freq)
    os.remove("trajectories_with_metrics.nc")
    res_trajects.to_netcdf("trajectories_with_metrics.nc")
    lp.print_stats()
    print("programm completed in" + str(round(time.time()-start,0)) + "s.")
    print("memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "Kb")
