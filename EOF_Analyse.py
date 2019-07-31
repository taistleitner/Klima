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
import pandas as pd
import time
import gc
import os
import datetime

os.environ["OMP_NUM_THREADS"] = "1" 

variable_mslp = ["PRMSL_GDS0_MSL","mslp"]
variable_rh = ["RH_GDS0_ISBL","rh"]
variable_spfh = ["SPFH_GDS0_ISBL","spfh"]
df_anom = xr.open_mfdataset('/home/srvx11/lehre/users/a1656041/ipython/Klima1/Anomalien/*.nc',combine='by_coords')
df_anom = df_anom.rename({variable_mslp[0]:variable_mslp[1],variable_rh[0]:variable_rh[1],variable_spfh[0]:variable_spfh[1]})
#print(df_anom)
df_anom = df_anom.sel(time = df_anom.time.dt.year.isin(np.arange(1961,2018,1)))
df_anom = df_anom.sel(time=~((df_anom.time.dt.month == 2) & (df_anom.time.dt.day == 29)))
df_anom = df_anom.chunk({'time':2000,'lat':29,'lon':29})
#print(df_anom)

df_date_range = pd.date_range(start='1961-01-01', end='2017-12-31', freq='D')
leapyears = np.arange(1964,2017,4)

leap = []
for each in df_date_range:
    if each.month==2 and each.day ==29:
        leap.append(each)

df_date_range = df_date_range.drop(leap)
#print(df_date_range)

df_norms = []
df_dates = []
TD_list = []

df_roll = df_anom.rolling(time=21,center=True).construct("window_dim")
df_roll = df_roll.sel(time=~((df_roll.time.dt.month == 2) & (df_roll.time.dt.day == 29)))
#print(df_roll.groupby("time.dayofyear"))
#print(df_roll.sel(time = (df_roll.time.dt.year.isin(1960) & df_roll.time.dt.dayofyear.isin(np.arange(55,65,1)))).time)
#df_roll_sel = df_anom_sel.rolling(time=21,center=True).construct("window_dim")

l = 1

#for day in df_roll.sel(time = df_roll.time.dt.dayofyear.isin(np.arange(58,64,1))).groupby("time.dayofyear"):
for day in df_roll.groupby("time.dayofyear"):
    if l == 366: break
    #EOF_dataset with all doys (including window)
    cube = day[1].sel(time=~((day[1].time.dt.month == 2) & (day[1].time.dt.day == 29))).rename({"time":
                                                                                               "time_old"}).stack(time = ("time_old",
                                                           "window_dim")).reset_index("time").transpose("time",
                                                                                                         "lat",
                                                                                                         "lon").dropna("time")
    
    #print(day[1].time)
    #print(cube.time)
    
    cube_mslp = cube.mslp.assign_coords(time=range(0,len(cube.time))).to_iris()
    cube_rh = cube.rh.assign_coords(time=range(0,len(cube.time))).to_iris()
    cube_spfh = cube.spfh.assign_coords(time=range(0,len(cube.time))).to_iris()
    solver = MultivariateEof([cube_mslp,cube_rh,cube_spfh],weights="coslat")
    
    doy = day[1].time.dt.dayofyear[0].values
    print('Processing doy # {}'.format(doy))
    #print(cube.time)
    
    doy_list = pd.to_datetime(day[1].time.values)
    #delta_t = pd.timedelta_range(start='0 day', periods=11, freq='D')
    reconstructed_time = doy_list
    
    for t in doy_list:
        #dplus = t+delta_t
        #dminus = t-delta_t
        #dummy_t = dminus.union(dplus)
        #reconstructed_time = reconstructed_time.union(dummy_t)
        
        tt = df_date_range.get_loc(t)
        
        tt_start = tt - 10
        if tt_start < 0: tt_start = 0
        tt_stop = tt + 10
        if tt_stop > (len(df_date_range)-1): tt_stop = len(df_date_range)-1
        
        dummy_period = pd.date_range(start=df_date_range[tt_start], end=df_date_range[tt_stop], freq='D')
        #print('dummy period for doy {}'.format(t))
        #print(dummy_period)
        reconstructed_time = reconstructed_time.union(dummy_period)
    
    rec_time_pool = reconstructed_time[reconstructed_time.isin(df_date_range)]
    
    leap2 = []
    for each2 in rec_time_pool:
        if each2.month==2 and each2.day ==29:
            leap2.append(each)

    rec_time_pool = rec_time_pool.drop(leap2)
    #print(rec_time_pool)
    
    var = 0
    i = 1
    var_list = []
    while var < 0.9:
        v = solver.varianceFraction(neigs=i)
        var = v.data.sum()
        var_list += [var]
        i += 1
    #print('Number of eofs = {}'.format(i))
    
    eofs = xr.DataArray.from_iris(solver.eofs(neofs=i)[0]).rename('eofs')
    pcs = xr.DataArray.from_iris(solver.pcs(npcs=i)).rename('pcs').drop('time_old').drop('window_dim').drop('dayofyear').assign_coords(time=rec_time_pool)
    evs = xr.DataArray.from_iris(solver.eigenvalues(neigs=i)).rename('evs')
    
    df_anom_doy = df_anom.sel(time = df_anom.time.dt.dayofyear.isin(doy))
    cube_doy_mslp = df_anom_doy.mslp.to_iris()
    cube_doy_rh = df_anom_doy.rh.to_iris()
    cube_doy_spfh = df_anom_doy.spfh.to_iris()
    
    ppcs = solver.projectField([cube_doy_mslp,cube_doy_rh,cube_doy_spfh], neofs=i)
    ppcs = xr.DataArray.from_iris(ppcs).rename('ppcs').drop('dayofyear')
    
    years = ppcs.time.dt.year.values
    
    dd = 0
    for j in years:
        #TD = datetime.datetime(j, 1, 1) + datetime.timedelta(doy - 1.)
        TD = doy_list[dd]
        
        if (j in leapyears) and (doy > 59):
            TD = TD + datetime.timedelta(days=1)
        
        print('Calculating norm for Target Day # {}'.format(TD))
        
        norm = ((ppcs.sel(time="{}".format(j))[0]-pcs)**2).sum(dim='pc')
        norm = norm.where(norm.time.dt.year != j, drop=True)
        analoga_dates = norm.sortby(norm).time.values[:10]
        analoga_norms = norm.sortby(norm).values[:10]
        
        df_norms += [analoga_norms]
        df_dates += [analoga_dates]
        TD_list += [TD]
        
        dd += 1
    
    
    np.savez_compressed("EOFs_Analyse_doy_{}.npz".format(doy),
         eofs=eofs,
         neofs=i,
         pcs=pcs,
         ppcs=ppcs,
         evs=evs,
         var=var_list)
    
    #print('l is {}'.format(l))
    print('Saved EOF file for doy # {}'.format(doy))
    
    l += 1
    del cube, cube_mslp, cube_rh, cube_spfh, solver, var, var_list, df_anom_doy, cube_doy_mslp, cube_doy_rh, cube_doy_spfh, ppcs, years, norm, analoga_dates, analoga_norms
    gc.collect()

da_norms = xr.DataArray(df_norms, coords=[TD_list, np.arange(1,11,1)], dims=['TD','rank']).sortby('TD')
da_dates = xr.DataArray(df_dates, coords=[TD_list, np.arange(1,11,1)], dims=['TD','rank']).sortby('TD')
ds_analoga = xr.Dataset({'analoga_dates': da_dates, 'analoga_norms': da_norms})
ds_analoga.to_netcdf('/home/srvx11/lehre/users/a1656041/ipython/Klima1/Analog_Method/Analoga_ranked_pro_TD.nc')

print('Saved analoga .nc File, closing')