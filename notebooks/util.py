"""This is a general purpose module containing routines
(a) that are used in multiple notebooks; or 
(b) that are complicated and would thus otherwise clutter notebook design.
"""

from __future__ import print_function
import re
import socket
import requests
import xml.etree.ElementTree as ET
import numpy
from xmip.preprocessing import combined_preprocessing

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

def load_ds_from_esgf_file_in_model_fnames_dict(model, model_fnames_dict, flg_onefile=False) : 
    
    ## Generate filename from model_fnames_dict
    fnames_i = model_fnames_dict[model]
    
    # Only open a single file
    if flg_onefile:
        fnames_i = [fnames_i[0]]

    # Open filenames
    ds_i = xr.open_mfdataset(fnames_i, combine='by_coords', compat='override', preprocess=combined_preprocessing)

    ## Subset by >50N
    cond_i = (ds_i['lat']>=50)
    dsnow = ds_i.where(cond_i,drop=True) #[[var_i]]
    
    return(dsnow)