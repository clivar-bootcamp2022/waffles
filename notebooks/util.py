"""This is a general purpose module containing routines
(a) that are used in multiple notebooks; or 
(b) that are complicated and would thus otherwise clutter notebook design.
"""

from __future__ import print_function
import re
import socket
import requests
import dask
import xarray as xr
import xmip.preprocessing as xmip
import numpy as np 


def is_ncar_host():
    """Determine if host is an NCAR machine."""
    hostname = socket.getfqdn()
    
    return any([re.compile(ncar_host).search(hostname) 
                for ncar_host in ['cheyenne', 'casper', 'hobart']])


# Author: Unknown
# I got the original version from a word document published by ESGF
# https://docs.google.com/document/d/1pxz1Kd3JHfFp8vR2JCVBfApbsHmbUQQstifhGNdc6U0/edit?usp=sharing

# API AT: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API#results-pagination

def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"]= "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = [] 
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose:
                for k in d:
                    print("{}: {}".format(k,d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)

def reindex_lat(ds):
    # check if lat is decreasing
    if ds.lat.isel(x=0,y=0) > 0:
        ds = ds.reindex(y=list(reversed(ds.y))).assign_coords(y=ds.y)
    
    return ds

def model_preproc(ds):
    # fix naming
    ds = xmip.rename_cmip6(ds)
    # reindex y if lat is decreasing
    ds = reindex_lat(ds)
    # promote empty dims to actual coordinates
    ds = xmip.promote_empty_dims(ds)
    # demote coordinates from data_variables
    ds = xmip.correct_coordinates(ds)
    # broadcast lon/lat
    ds = xmip.broadcast_lonlat(ds)
    # shift all lons to consistent 0-360
    ds = xmip.correct_lon(ds)
    # fix the units
    ds = xmip.correct_units(ds)
    # rename the `bounds` according to their style (bound or vertex)
    ds = xmip.parse_lon_lat_bounds(ds)
    # sort verticies in a consistent manner
    ds = xmip.sort_vertex_order(ds)
    # convert vertex into bounds and vice versa, so both are available
    ds = xmip.maybe_convert_bounds_to_vertex(ds)
    ds = xmip.maybe_convert_vertex_to_bounds(ds)
    ds = xmip.fix_metadata(ds)
    ds = ds.drop_vars(["bnds", "vertex"], errors="ignore")
    return ds

def load_ds_from_esgf_file_in_model_fnames_dict(model, model_fnames_dict, flg_onefile=False):    
    ## Generate filename from model_fnames_dict
    fnames_i = model_fnames_dict[model]
    
    # Only open a single file
    if flg_onefile=='True':
        fnames_i = [fnames_i[0]]

    # Open filenames
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = xr.open_mfdataset(fnames_i, combine='by_coords',
                               compat='override', preprocess=model_preproc) #.persist()
    
    # Subset by >50N
    cond = (ds['lat']>=50)
    dsnow = ds.where(cond,drop=True) #.persist()
    
    # rechunk
    dsnow = dsnow.chunk(chunks={'x':50,'y':50}) #.persist() #'time':-1,'lev':-1
    
    return(dsnow)

def calc_Bering_fluxes(DS):
    # Model reference density [kg/m3]
    rho_0 = 1035
    # Reference potential temperature for heat flux [deg C]
    theta_ref = -1.9
    # Heat capacity of water for model output [J/kg K]
    C_p = 3992
    # Reference salinity for freshwater flux [PSU]
    S_ref = 34.8
    
    # Volume transport [m3/s]
    DS['T_vol'] = DS.vmo/rho_0
    
    # Heat flux [J/s]
    DS['F_heat'] = rho_0 * DS.T_vol * C_p * (DS.thetao - theta_ref)
    
    # Freshwater flux [km3/s]
    DS['F_fresh'] = DS.T_vol * (1 - (DS.so/S_ref)) * (10**-9)

    return DS


def calc_dpe(DS,H=500,norm=2):
    '''
    Calculates \Delta PE, as in Muilwijk et al (2022) [https://eartharxiv.org/repository/view/3361/]
    from the surface to the target depth H, but normalizes with D**norm, where D is the maximum of 
    H and the ocean depth.
    
    NOTE: Assumes positive z-coordinate, increasing downward
    '''
    g = 9.81 # gravitational acceleration
    
    # select upper H meter (H should be below halocline for all models, > 300m
    DS = DS.sel(lev=slice(0,H))
    print('Lowest level: '+str(DS.lev_bounds.isel(bnds=1).values[-1]))
    # calculate potential density
    DS['rho'] = xr.apply_ufunc(gsw.sigma0,DS.so,DS.thetao,dask='parallelized')

    # calculate ds for integration and weighted mean
    DS['dz'] = (('lev'), DS.lev_bounds.isel(bnds=1).values - DS.lev_bounds.isel(bnds=0).values)

    #DS['levv'] = ((DS.rho*0 +1)*DS.lev_bounds.isel(bnds=1)).max(dim='lev')
    DS['levv'] = xr.apply_ufunc(np.minimum,DS['deptho'],
                                DS['lev_bounds'].sel(bnds=-1),dask='parallelized').isel(lev=-1)
     
    if norm == 0:
        factor = 1
    elif norm == 1:
        factor = DS.levv
    elif norm == 2:
        factor = DS.levv**2
        
    # calculate tempeterature, salinity and density of a fully mixed water column
    DS['T_mix']   = DS.thetao*0 + DS.thetao.weighted(weights=DS.dz).mean(dim='lev')
    DS['S_mix']   = DS.so*0 + DS.so.weighted(weights=DS.dz).mean(dim='lev')
    DS['rho_mix'] = xr.apply_ufunc(gsw.sigma0,DS.S_mix,DS.T_mix,dask='parallelized') 

    # calculate DPE according to Muiwijk et al. (2022)
    DS['pe']     = g * (DS.rho*    DS.lev).weighted(weights=DS.dz).sum(dim='lev') 
    DS['pe_mix'] = g * (DS.rho_mix*DS.lev).weighted(weights=DS.dz).sum(dim='lev') 
    DS['dpe']    = (DS.pe - DS.pe_mix ) / factor
    
    # return the DataSet and the actual H-value in that model
    return DS,int(np.round(DS.lev_bounds.isel(bnds=1).values[-1]))

def subset_model_by_lat_ind(dsnow, dsnow_gt_bs_lat_i) : 
    #Calc all min and max x and y indexes
    yind1 = dsnow_gt_bs_lat_i.lat.isel(y=0).y.values
    yind2 = dsnow_gt_bs_lat_i.lat.isel(y=-1).y.values
    xind1 = dsnow_gt_bs_lat_i.lat.isel(x=0).x.values
    xind2 = dsnow_gt_bs_lat_i.lat.isel(x=-1).x.values

    ## Determine if x or y index corresponds to lat and if min or max index corresponds to lat

    y1_ind_diff = np.nanmax(dsnow_gt_bs_lat_i.lat.sel(y=yind1).values)-np.nanmin(dsnow_gt_bs_lat_i.lat.sel(y=yind1).values)
    y2_ind_diff = np.nanmax(dsnow_gt_bs_lat_i.lat.sel(y=yind2).values)-np.nanmin(dsnow_gt_bs_lat_i.lat.sel(y=yind2).values)

    x1_ind_diff = np.nanmax(dsnow_gt_bs_lat_i.lat.sel(x=xind1).values)-np.nanmin(dsnow_gt_bs_lat_i.lat.sel(x=xind1).values)
    x2_ind_diff = np.nanmax(dsnow_gt_bs_lat_i.lat.sel(x=xind2).values)-np.nanmin(dsnow_gt_bs_lat_i.lat.sel(x=xind2).values)

    # First test if lat range corresponding to xind or yind max-min is bigger
    if (y1_ind_diff < x1_ind_diff) & (y2_ind_diff < x2_ind_diff)  : 
        # If true then y ind corresponds to determining lat 
        # Ideally should test this for both x1 and x2 BUT should be same

        #Test if y1 or y2 corresponds to min lat
        if np.nanmin(dsnow_gt_bs_lat_i.lat.sel(y=yind1).values) <= np.nanmin(dsnow_gt_bs_lat_i.lat.sel(y=yind2).values) : 
            dsnow_bs_lat_i_ind = dsnow.sel(y=yind1) #dsnow_gt_bs_lat_i.sel(y=yind1)

        else : 
            print("y2 > y1")
            dsnow_bs_lat_i_ind = dsnow.sel(y=yind2) #dsnow_gt_bs_lat_i.sel(y=yind2)


    elif (y1_ind_diff > x1_ind_diff) & (y2_ind_diff > x2_ind_diff)  : 
        # If True then x ind corresponds to determining lat
        print('Note : x ind seems to correspond to lat not y (as expected)!')

        #Test if y1 or y2 corresponds to min lat
        if np.nanmin(dsnow_gt_bs_lat_i.lat.sel(x=xind1).values) < np.nanmin(dsnow_gt_bs_lat_i.lat.sel(x=xind2).values) : 
            dsnow_bs_lat_i_ind = dsnow.sel(x=xind1) #dsnow_gt_bs_lat_i.sel(x=xind1)

        else : 
            print("y2 > y1")
            dsnow_bs_lat_i_ind = dsnow.sel(x=xind2) #dsnow_gt_bs_lat_i.sel(x=xind2)

    else : 
        print("Something weird is going on - check what is happening. x1_ind_diff is : "+str(x1_ind_diff)+", x2_ind_diff is : "+str(x2_ind_diff)+"y1_ind_diff is : "+str(y1_ind_diff)+", y2_ind_diff is : "+str(y2_ind_diff))
        dsnow_bs_lat_i_ind = []
              
    return(dsnow_bs_lat_i_ind)

def subset_ds_bering_trans(dsnow, model_name, lat_bs_i, bering_minlon, bering_maxlon) :
    #Subset by bs 
    cond_bs_lat_i = (dsnow['lat']>=lat_bs_i) & (dsnow['lon']>=bering_minlon) & (dsnow['lon']<=bering_maxlon)
    dsnow_gt_bs_lat_i = dsnow.where(cond_bs_lat_i ,drop=True) #[[var_i]]

    # Subset models by lat index
    dsnow_bs_lat_i_ind = subset_model_by_lat_ind(dsnow, dsnow_gt_bs_lat_i)

    #print('I made it this far.')

    ## Subselect lat by edges of land (depth=0) on either side 
    dsnow_bs_lat_i_ind = dsnow_bs_lat_i_ind.where(dsnow_bs_lat_i_ind.lon>bering_minlon-5, 
                                                  drop=True).where(dsnow_bs_lat_i_ind.lon<bering_maxlon+5, drop=True)

    #Mask for where deptho is 0 /nan
    dsnow_bs_lat_i_where0 = (dsnow_bs_lat_i_ind.deptho>0)

    #Find left and right indexes of straight
    ileft = dsnow_bs_lat_i_where0.idxmax().values-1 #search from left for index of first True
    dsnow_bs_lat_i_where0 = dsnow_bs_lat_i_where0.sortby('x', ascending=False)
    iright = dsnow_bs_lat_i_where0.idxmax().values+1 #search from right for index of first True        
    #print indexes and lon values
    print(model_name, ileft, iright, " IE ", dsnow_bs_lat_i_ind.sel(x=ileft).lon.values, "-", dsnow_bs_lat_i_ind.sel(x=iright).lon.values)
    
    dsnow_bs_lat_i_ind = dsnow_bs_lat_i_ind.sel(x=slice(ileft, iright))
    
    return(dsnow_bs_lat_i_ind)