"""
Computation of cloudmetrics along interpolated ERA5 trajectories.
"""
import sys
import os

# Get the parent directory and add it to sys.path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

import cloudmetrics.cloudmetrics as clmt
import numpy as np
import xarray as xr
#import cloudmetrics
import time 
from datetime import datetime
import glob
import os
import resource
from line_profiler import LineProfiler

from utils import extract_frame, get_centered_window

def main():
    # set inputs
    data_dir = "/scratch-shared/rmeier/Data/GOES-CMIP-C13-Tropical-North-Atlantic/daily/"
    ref_dir = "/home/rmeier/Data/goes16_reference/"
    traj_dir = "/home/rmeier/Data/Trajectories/NAtl_Trajectories_Mid_Start_925hPa_1hrLocalInterp_ERA5_vars_Dec-Feb_2018-2022_subsets/"
    traj_file_name = "subset_0" # without .nc
    save_freq = 100 # save after every 100 images
    framesize = 5
    accessmode= "netCDF" 
    comp_all_metrics = False
    profiler = False

    start = time.time()

    if profiler:
        #initialize line profiler
        lp = LineProfiler()
        lp_wrapper = lp(compute_metrics)

    # get datasets
    trajects = xr.open_dataset(traj_dir + traj_file_name + ".nc")
    goes_ref_ds = xr.open_dataset(ref_dir + "goes_ref_ds.nc")
    
    print(str(datetime.now())+": Selecting trajectory extent...")
    traj_extents = (-70+framesize/2,-10-framesize/2,0+framesize/2,40-framesize/2) 
    trajects = trajects.where((trajects.longitude >= traj_extents[0])
                             &(trajects.longitude <= traj_extents[1])
                             &(trajects.latitude >= traj_extents[2])
                             &(trajects.latitude <= traj_extents[3]))  
    
    # compute
    if profiler:
        res_trajects = lp_wrapper(trajects,goes_ref_ds,data_dir,traj_dir,traj_file_name,framesize,save_freq,accessmode,comp_all_metrics)
        os.remove(traj_dir + traj_file_name + "_with_metrics.nc")
        res_trajects.to_netcdf(traj_dir + traj_file_name + "_with_metrics.nc")
        lp.print_stats()
    else:
        res_trajects = compute_metrics(trajects,goes_ref_ds,data_dir,traj_dir,traj_file_name,framesize,save_freq,accessmode,comp_all_metrics)
        os.remove(traj_dir + traj_file_name + "_with_metrics.nc")
        res_trajects.to_netcdf(traj_dir + traj_file_name + "_with_metrics.nc")
 
    print("programm completed in" + str(round(time.time()-start,0)) + "s.")
    print("memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "Kb")


def compute_metrics(trajects,goes_ref_ds,data_dir,traj_dir,traj_file_name,framesize=5,save_freq=100,accessmode="netCDF",comp_all_metrics=False):
    """
    Computes metrics on interpolated trajectories and gives out the trajectory dataset with metrics.

    Parameters
    ---------------------------------------------------------------------------------------
    - trajects: xarray.Dataset with dimensions "N_Trajectories" and "Time"
        Interpolated trajectories with latitude, longitude and UTC time along 
        the "Time" dimension
    - goes_ref_ds: xarray.Dataset with dimension "time"
        Dataset of GOES-16 images of winter seasons (DJF) between 2017/18 and 2022/23 
        with start-, end- and centraltime of the scan and datestring and t_index to 
        allocate the image in downloaded files
    - data_dir: str        
        Directory, where image files are stored
    - traj_dir: str        
        Directory, where trajectory files are stored
    - traj_file_name: str
        File name of the trajectory dataset  
    - framesize: int
        Size of the square lat/lon frame in degree, where the metrics are computed on 
        (standard 5x5)
    - accessmode: "netCDF" or "json" or "no access"
        Mode of access via downloaded netCDF files or via json mapping to AWS server, 
        where GOES-16 data is stored. Currently json mode is not implemented! 
        Choose "no access" for test run without accessing image files.

    Return
    ---------------------------------------------------------------------------------------
    - trajects: xarray.Dataset with dimensions "N_Trajectories" and "Time"
        Same dataset as input, but metrics added as variables along the "Time" dimension
    """
    print(str(datetime.now())+": Initialize cloud metrics...")

    # Get number and maximum size of the trajectories
    N_Trajectories = trajects.sizes["N_Trajectories"]
    N_timesteps = trajects.sizes["Time"]
    N_masks = 2
    
    # initialize metric arrays 
    
    # comparisons with Master thesis & Bony et al.
    l_mean_comp = np.full((N_timesteps,N_Trajectories),np.nan)
    cloud_frac_comp = np.full((N_timesteps,N_Trajectories),np.nan)
    hcf_comp = np.full((N_timesteps,N_Trajectories),np.nan)
    iorg_comp = np.full((N_timesteps,N_Trajectories),np.nan)

    # high cloud fraction
    hcf = np.full((N_timesteps,N_Trajectories),np.nan)

    # scalar statistical metrics
    mean_BT = np.full((N_timesteps,N_Trajectories),np.nan)
    var_BT = np.full((N_timesteps,N_Trajectories),np.nan)
    BT_5_perc = np.full((N_timesteps,N_Trajectories),np.nan)
    if comp_all_metrics:
        skew_BT = np.full((N_timesteps,N_Trajectories),np.nan)
        kurt_BT = np.full((N_timesteps,N_Trajectories),np.nan)
    
    # scalar spectral metrics
    spec_len_median = np.full((N_timesteps,N_Trajectories),np.nan)
    spec_len_moment = np.full((N_timesteps,N_Trajectories),np.nan)

    # cloud mask metrics
    cloud_frac = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    open_sky_max = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    open_sky_perc = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    open_sky_mean = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    #open_sky_rad = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    cloud_depth_est = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    fractal_dim = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)

    # object based metrics
    l_max = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    l_mean = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    num_objects = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    iorg = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    if comp_all_metrics:
        scai = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
        mean_perimeter = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
        cop = np.full((N_timesteps,N_Trajectories,N_masks),np.nan)
    
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
            t_index = goes_ref_ds.sel(time=goes_time).t_index.values            # deprecated

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
                
                # compute cloud masks
                CMI = CMIP.CMI.values
                gauss_cloud_thresh = traj.skt.values - 6.7 
                cloud_upper_thresh = [gauss_cloud_thresh,gauss_cloud_thresh-2.5,290] 
                cloud_lower_thresh = [270,270,280] 
                
                finite_mask = np.isfinite(CMI)
                no_cirrus_mask = finite_mask * (CMI>270)

                masks = []
                for j in range(N_masks+1):
                    mask = np.where(finite_mask,
                                    (CMI < cloud_upper_thresh[j])*(CMI > cloud_lower_thresh[j]),
                                    np.nan)
                    masks.append(mask) 
                
                # compute the metrics
                if np.sum(no_cirrus_mask) == 0:
                    # only high clouds
                    hcf[time_indices[i],traj_indices[i]] = 1
                    hcf_comp[time_indices[i],traj_indices[i]] = 1
                    continue
                else:
                    # high cloud fraction
                    hcf[time_indices[i],traj_indices[i]] = clmt.scalar.high_cloud_fraction(CMI,cloud_lower_thresh[0])
                    hcf_comp[time_indices[i],traj_indices[i]] = clmt.scalar.high_cloud_fraction(CMI,280)

                    # BT statistics (Cirrus excluded)
                    mean_BT[time_indices[i],traj_indices[i]] = clmt.scalar.mean(CMI,mask=no_cirrus_mask)
                    var_BT[time_indices[i],traj_indices[i]] = clmt.scalar.var(CMI,mask=no_cirrus_mask)
                    BT_5_perc[time_indices[i],traj_indices[i]] = clmt.scalar.perc(CMI,5,mask=no_cirrus_mask)
                    if comp_all_metrics:
                        skew_BT[time_indices[i],traj_indices[i]] = clmt.scalar.skew(CMI,mask=no_cirrus_mask)
                        kurt_BT[time_indices[i],traj_indices[i]] = clmt.scalar.kurtosis(CMI,mask=no_cirrus_mask)

                    # compute comparison metrics
                    object_mask = np.where(np.isfinite(masks[2]),masks[2],0)

                    cloud_frac_comp[time_indices[i],traj_indices[i]] = clmt.mask.cloud_fraction(mask=mask[2])
                    l_mean_comp[time_indices[i],traj_indices[i]] = clmt.mask.mean_object_length_scale(mask=object_mask,periodic_domain=False)
                    iorg_comp[time_indices[i],traj_indices[i]] = clmt.mask.iorg_objects(mask=object_mask,periodic_domain=False)


                # apply square window to scalar field for spectral metric computation
                CMI_centered = get_centered_window(CMI,int(min(CMI.shape)*0.5/2)*2)

                # compute spectral metrics
                spec_len_moment[time_indices[i],traj_indices[i]] = clmt.scalar.compute_spectral_length_moment(CMI_centered)

                for j in range(N_masks):
                    cloud_depth_est[time_indices[i],traj_indices[i],j] = (cloud_upper_thresh[j] - clmt.scalar.perc(CMI,5,mask=no_cirrus_mask)) / 5
                    cloud_frac[time_indices[i],traj_indices[i],j] = clmt.mask.cloud_fraction(mask=masks[j])
                    open_sky = clmt.mask.open_sky_stats(mask=masks[j])
                    open_sky_max[time_indices[i],traj_indices[i],j] = open_sky[0]
                    open_sky_perc[time_indices[i],traj_indices[i],j] = open_sky[1]
                    open_sky_mean[time_indices[i],traj_indices[i],j] = open_sky[2]
                    #open_sky_rad[time_indices[i],traj_indices[i],j] = clmt.mask.open_sky_rad(mask=masks[j])
                    fractal_dim[time_indices[i],traj_indices[i],j] = clmt.mask.fractal_dimension(mask=masks[j])

                    object_mask = np.where(np.isfinite(masks[j]),masks[j],0)
                    if len(np.argwhere(object_mask==1)) > 0:
                        l_max[time_indices[i],traj_indices[i],j] = clmt.mask.max_object_length_scale(mask=object_mask,periodic_domain=False)
                        l_mean[time_indices[i],traj_indices[i],j] = clmt.mask.mean_object_length_scale(mask=object_mask,periodic_domain=False)
                        num_objects[time_indices[i],traj_indices[i],j] = clmt.mask.num_objects(mask=object_mask,periodic_domain=False)
                        iorg[time_indices[i],traj_indices[i],j] = clmt.mask.iorg_objects(mask=object_mask,periodic_domain=False)
                        if comp_all_metrics:
                            scai[time_indices[i],traj_indices[i],j] = clmt.mask.scai_objects(mask=object_mask,periodic_domain=False)
                            cop[time_indices[i],traj_indices[i],j] = clmt.mask.cop_objects(mask=object_mask,periodic_domain=False)
                            mean_perimeter[time_indices[i],traj_indices[i],j] = clmt.mask.mean_object_perimeter_length(mask=object_mask,periodic_domain=False)
        
        if (goes_index%save_freq) == 0:
            print(str(datetime.now())+": Saving...")
            res_trajects = trajects.assign(variables=dict(
                                                mean_BT=(["Time","N_Trajectories"],mean_BT),
                                                var_BT=(["Time","N_Trajectories"],var_BT),
                                                BT_5_perc=(["Time","N_Trajectories"],BT_5_perc),
                                                spec_len_median=(["Time","N_Trajectories"],spec_len_median),
                                                spec_len_moment=(["Time","N_Trajectories"],spec_len_moment),
                                                hcf=(["Time","N_Trajectories"],hcf),
                                                hcf_comp=(["Time","N_Trajectories"],hcf_comp),
                                                cloud_frac_comp=(["Time","N_Trajectories"],cloud_frac_comp),
                                                l_mean_comp=(["Time","N_Trajectories"],l_mean_comp),
                                                iorg_comp=(["Time","N_Trajectories"],iorg_comp),
                                                cloud_fraction=(["Time","N_Trajectories","Mask"],cloud_frac),
                                                open_sky_max=(["Time","N_Trajectories","Mask"],open_sky_max),
                                                open_sky_perc=(["Time","N_Trajectories","Mask"],open_sky_perc),
                                                open_sky_mean=(["Time","N_Trajectories","Mask"],open_sky_mean),
                                                #open_sky_rad=(["Time","N_Trajectories","Mask"],open_sky_rad),
                                                cloud_depth_est=(["Time","N_Trajectories","Mask"],cloud_depth_est),
                                                fractal_dim=(["Time","N_Trajectories","Mask"],fractal_dim),
                                                l_max=(["Time","N_Trajectories","Mask"],l_max),
                                                l_mean=(["Time","N_Trajectories","Mask"],l_mean),
                                                num_objects=(["Time","N_Trajectories","Mask"],num_objects),
                                                iorg=(["Time","N_Trajectories","Mask"],iorg)
                                                )
                                            )
            if comp_all_metrics:
                res_trajects = res_trajects.assign(variables=dict(
                                                skew_BT=(["Time","N_Trajectories"],skew_BT),
                                                kurt_BT=(["Time","N_Trajectories"],kurt_BT),     
                                                scai=(["Time","N_Trajectories","Mask"],scai),
                                                cop=(["Time","N_Trajectories","Mask"],cop),
                                                mean_perimeter=(["Time","N_Trajectories","Mask"],mean_perimeter)
                                                )
                                            )
                
            if len(glob.glob(traj_dir + traj_file_name + "_with_metrics.nc")) > 0:
                os.remove(traj_dir + traj_file_name +"_with_metrics.nc")
            res_trajects.to_netcdf(traj_dir + traj_file_name +"_with_metrics.nc")

        print(str(datetime.now())+": " +str(int((goes_index+1)/len(goes_ref_ds.datestring)*100)) + "% done",end="\r")
        
    res_trajects = trajects.assign(variables=dict(
                                                mean_BT=(["Time","N_Trajectories"],mean_BT),
                                                var_BT=(["Time","N_Trajectories"],var_BT),
                                                BT_5_perc=(["Time","N_Trajectories"],BT_5_perc),
                                                spec_len_median=(["Time","N_Trajectories"],spec_len_median),
                                                spec_len_moment=(["Time","N_Trajectories"],spec_len_moment),
                                                hcf=(["Time","N_Trajectories"],hcf),
                                                hcf_comp=(["Time","N_Trajectories"],hcf_comp),
                                                cloud_frac_comp=(["Time","N_Trajectories"],cloud_frac_comp),
                                                l_mean_comp=(["Time","N_Trajectories"],l_mean_comp),
                                                iorg_comp=(["Time","N_Trajectories"],iorg_comp),
                                                cloud_fraction=(["Time","N_Trajectories","Mask"],cloud_frac),
                                                open_sky_max=(["Time","N_Trajectories","Mask"],open_sky_max),
                                                open_sky_perc=(["Time","N_Trajectories","Mask"],open_sky_perc),
                                                open_sky_mean=(["Time","N_Trajectories","Mask"],open_sky_mean),
                                                #open_sky_rad=(["Time","N_Trajectories","Mask"],open_sky_rad),
                                                cloud_depth_est=(["Time","N_Trajectories","Mask"],cloud_depth_est),
                                                fractal_dim=(["Time","N_Trajectories","Mask"],fractal_dim),
                                                l_max=(["Time","N_Trajectories","Mask"],l_max),
                                                l_mean=(["Time","N_Trajectories","Mask"],l_mean),
                                                num_objects=(["Time","N_Trajectories","Mask"],num_objects),
                                                iorg=(["Time","N_Trajectories","Mask"],iorg)
                                                )
                                            ) 
    if comp_all_metrics:
        res_trajects = res_trajects.assign(variables=dict(
                                        skew_BT=(["Time","N_Trajectories"],skew_BT),
                                        kurt_BT=(["Time","N_Trajectories"],kurt_BT),     
                                        scai=(["Time","N_Trajectories","Mask"],scai),
                                        cop=(["Time","N_Trajectories","Mask"],cop),
                                        mean_perimeter=(["Time","N_Trajectories","Mask"],mean_perimeter)
                                        )
                                    )    
        
    return res_trajects


if __name__ == "__main__":
    main()
