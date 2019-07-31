import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.signal import detrend
from eofs.xarray import Eof
import iris
from eofs.multivariate.iris import MultivariateEof
import iris.quickplot as qplt
from tempfile import TemporaryFile
import time
import gc

variable_mslp = ["PRMSL_GDS0_MSL","mslp"]
variable_rh = ["RH_GDS0_ISBL","rh"]
variable_spfh = ["SPFH_GDS0_ISBL","spfh"]
pfad = "/home/srvx11/lehre/users/a1656041/ipython/Klima1/Data/JRA-55/all_datasets"
#pfad = "/home/srvx11/lehre/users/a1656041/ipython/Klima1/Data/JRA-55/mslp_daily"
#pfad = "/home/srvx11/lehre/users/a1656041/ipython/Klima1/Data/Data"

def open_df(pfad,var):
    df = xr.open_mfdataset(pfad+'*'+var+'*')
    df = df.rename({'initial_time0_hours':'time','g0_lat_1':'lat','g0_lon_2':'lon'}).drop("initial_time0_encoded")
    df.coords['lon'] = (df.coords['lon'] + 180) % 360 - 180
    df = df.chunk({'time':508,'lat':29,'lon':29})
    return df
def pref_df(ds_prep,window_size):
        ds_prep_resample = ds_prep.resample(time='1D').mean().chunk({'time':508,'lat':29,'lon':29})
        print("DailyMean_fertig")
        #ds_prep = ds_prep.chunk({'time':-1})
        ds_prep_roll = ds_prep.rolling(time=window_size, center=True).construct('window_dim')
        print("RollMean_fertig")
        return ds_prep_resample,ds_prep_roll
def anom_df(ds_prep,ds_prep_roll):
        ds_prep_clim = ds_prep_roll.groupby('time.dayofyear').mean(dim=['window_dim','time'])
        ds_prep_std = ds_prep_roll.groupby('time.dayofyear').std(dim=xr.ALL_DIMS)
        ds_prep = ds_prep.groupby('time.dayofyear') - ds_prep_clim
        ds_prep = ds_prep.groupby('time.dayofyear') / ds_prep_std
        #ds_prep = ds_prep.chunk({'time': -1})
        return ds_prep

df_mslp = open_df(pfad,'prmsl')
df_mslp.rename({variable_mslp[0]:variable_mslp[1]}).drop("initial_time0")#,variable_rh[0]:variable_rh[1],variable_spfh[0]:variable_spfh[1]}).drop("initial_time0")
df_mslp_resample,df_mslp_roll = pref_df(df_mslp,21)
print ("prep_df fertig")
df_mslp = anom_df(df_mslp_resample,df_mslp_roll)
df_mslp.to_netcdf('Anomalien_mslp.nc')
del df_mslp,df_mslp_resample,df_mslp_roll
gc.collect()

df_rh = open_df(pfad,'rh')
df_rh.rename({variable_rh[0]:variable_rh[1]}).drop("initial_time0")
df_rh_resample,df_rh_roll = pref_df(df_rh,21)
print ("prep_df fertig")
df_rh = anom_df(df_rh_resample,df_rh_roll)
df_rh.to_netcdf('Anomalien_rh.nc')
del df_rh,df_rh_resample,df_rh_roll
gc.collect()

df_spfh = open_df(pfad,'spfh')
df_spfh.rename({variable_spfh[0]:variable_spfh[1]}).drop("initial_time0")
df_spfh_resample,df_spfh_roll = pref_df(df_spfh,21)
print ("prep_df fertig")
df_spfh = anom_df(df_spfh_resample,df_spfh_roll)
df_spfh.to_netcdf('Anomalien_spfh.nc')
gc.collect()

df = xr.open_mfdataset('*.nc')
print(df)