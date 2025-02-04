    #!/usr/bin/env python
    # coding: utf-8

    # # Import packages

    # In[1]:

import argparse

def build_model(path):

    import os
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from datetime import datetime
    import matplotlib.pyplot as plt

    from hydromt_sfincs import SfincsModel
    from hydromt.config import configread
    from hydromt_sfincs import utils
    from hydromt.log import setuplog


    # # Functions

    # In[2]:


    # Compute river depth using Manning equation

    from scipy.optimize import newton

    def h_newton(gdf_riv):
        # Constants
        B = gdf_riv['rivwth']
        n = gdf_riv['manning']
        S = gdf_riv['slp']
        K = B**(5/3)*np.sqrt(S)/(3*n)
        Q = gdf_riv['qbankfull']

        # Define the function and its derivative
        def f(h, B, K, Q):
            return K * h**(5/3) / (h + 2*B)**(2/3) - Q

        def df(h, B, K, Q):  # Add Q as an argument
            return K * h**(2/3) * (6*h + 5*B) / (2*h + B)**(5/3)

        # Initialize the river depth column
        gdf_riv['rivdph'] = np.nan
        gdf_riv['k'] = K

        # Apply the Newton-Raphson method to each row
        for i, row in gdf_riv.iterrows():
            B = row['rivwth']
            K = row['k']
            Q = row['qbankfull']  # Extract the scalar value
            h0 = 1  # initial guess
            h = newton(f, h0, df, args=(B, K, Q), maxiter=1000)
            gdf_riv.at[i, 'rivdph'] = np.round(h,2)

        return gdf_riv[['geometry','rivwth', 'qbankfull','slp', 'rivdph','manning']]


    # # Build model

    # In[3]:


    model_file = configread(path, abs_path=True)

    root = str(model_file['root']) # Model name
    data_libs = str(model_file['data_libs']) # Data catalog path
    region = str(model_file['region'])
    logger = setuplog(root, log_level = 10) # Show messages when executing model operations

    sf = SfincsModel(root=root, mode='w+', data_libs=data_libs) # Initialize model class -> Create model folder


    # # Setup grid

    # In[4]:

    sf.setup_grid_from_region(region={'geom': region}, res=model_file['res'], crs=str(model_file['crs']))


    # # Computational parameters

    # In[5]:


    sf.set_config('tref', str(model_file['tref']))
    sf.set_config('tstart', str(model_file['tstart']))
    sf.set_config('tstop', str(model_file['tstop']))
    sf.set_config('dtmax', model_file['dtmax'])
    sf.set_config('alpha', str(model_file['alpha']))


    # # Mask for computational mesh

    # In[6]:


    sf.setup_mask_active(mask=str(model_file['mask_active']))
    sf.setup_mask_bounds(btype='outflow', include_mask=str(model_file['mask_bounds']))


    # # Setup depth

    # In[7]:


    elevtn = sf.data_catalog.get_rasterdataset(str(model_file['elevtn'])) # Read DEM
    datasets_dep = [{'elevtn': elevtn}]
    sf.setup_dep(datasets_dep=datasets_dep)


    # # Compute river bathymetry

    # In[8]:


    # Add slope to river segments where slope is 0

    rivers = sf.data_catalog.get_geodataframe(str(model_file['rivers']), geom = sf.region)
    rivers = rivers[rivers['rivwth'] > 100] # Extract those rivers wider than 100 meters
    rivers['manning'] = 0.02 # Set a default Manning roughness of 0.2
    rivers = rivers[['geometry','rivwth', 'qbankfull','slp', 'manning']]

    # Check if there are any zero values in the 'slp' column
    if (rivers['slp'] == 0).any():
        slp_min = rivers[rivers['slp']!=0]['slp'].min()
        rivers['slp'] = rivers['slp'].replace(0, slp_min)
        print("The value 0  was found in the slope of a segment.")
        print("The new slope value is:", slp_min)
    else:
        print("There are no zero values in the 'slp' column.")


    # In[9]:


    # Compute river depth per river segment using Manning equation

    rivers = h_newton(rivers)
    rivers['rivdph'] = rivers['rivdph'] + 1.8 # Calibration parameter: Increase river depth to reduce river level
    sf.geoms['rivers_inflow'] = rivers # Add river geometries to model
    riv_fn = '/Users/aprida/Documents/Consulting/Private_sector/Keolis/Model_Alvaro/Inputs/rivers_inflow_modified.geojson'
    rivers.to_file(riv_fn)


    # # Setup Manning roughness

    # In[10]:


    gdf_riv = sf.geoms['rivers_inflow']
    datasets_riv = [{'centerlines': gdf_riv}] # To be used in Setup_subgrid for bathymetry
    gdf_riv_buf = gdf_riv.assign(geometry=gdf_riv.buffer(gdf_riv['rivwth']/3)) # Generate river polygon using river width buffer
    da_manning = sf.grid.raster.rasterize(gdf_riv_buf, 'manning', nodata=np.nan, all_touched=True) # Rasterize Manning roughness in river polygon
    datasets_rgh = [{'manning': da_manning}, {'lulc': str(model_file['lulc'])}] # Overlay Manning roughness in river polygon and in dry areas (application of Vito LULC as default)

    #sf.setup_manning_roughness(datasets_rgh=datasets_rgh)
    #fig, ax = sf.plot_basemap(variable="dep", plot_bounds=False, bmap="sat", zoomlevel=12, plot_geoms=False)


    # # Setup subgrid

    # In[11]:


    dem_res = np.round(elevtn.rio.reproject(sf.crs).raster.res[0])
    grid_res = np.round(sf.res[0])
    sub_res = int(grid_res/dem_res) # Subgrid resolution ratio is set to get the higher resolution
                                        #If grid_res = 300 , dem_res = 30 ==> sub_res  = 10 (int(300/30))


    # In[12]:


    sf.setup_subgrid(datasets_dep=datasets_dep,
                    datasets_rgh=datasets_rgh,
                    datasets_riv=datasets_riv,
                    nr_subgrid_pixels=sub_res,
                    write_dep_tif= True,
                    write_man_tif= True) # Set up subgrid as a stack of the different raster inputs (DEM, roughness, etc.)

    #This line stores the subgrid dem properly as a raster file in root/subgrid/dep_subgrid.tif
    sf.data_catalog.get_rasterdataset(os.path.join(root, 'subgrid/dep_subgrid.tif')).rio.to_raster(os.path.join(root, 'subgrid/dep_subgrid.tif'))


    # # Setup discharge

    # In[13]:


    if 'bnd' in model_file.keys():
    
        dis = pd.read_csv(str(model_file['bnd']), index_col='datetime')
        index = dis.columns
        
        tstart_dis = datetime.strptime(dis.index[0], '%Y-%m-%d %H:%M:%S')
        tend_dis = datetime.strptime(dis.index[-1], '%Y-%m-%d %H:%M:%S')
        tref_dis = tstart_dis
        
        sf.set_config("tref", tref_dis.strftime('%Y%m%d %H%M%S'))
        sf.set_config("tstart", tstart_dis.strftime('%Y%m%d %H%M%S'))
        sf.set_config("tstop", tend_dis.strftime('%Y%m%d %H%M%S'))
        
        dis = dis.values
        
        time = pd.date_range(
        start = utils.parse_datetime(sf.config["tstart"]),
        end = utils.parse_datetime(sf.config["tstop"]),freq = 'H')
        
        dispd = pd.DataFrame(index=time, columns=index, data=dis).fillna(method='ffill')
        
        src = gpd.read_file(str(model_file['src']))
        src = src.set_index('index')[['geometry']]
        
        sf.setup_discharge_forcing(timeseries=dispd, locations=src)

    else:
        
        pass


    # # Setup precipitation

    # In[14]:


    # 2D precipitation

    # sf.setup_precip_forcing_from_grid(precip='precip_era5_hourly', aggregate=False)

    # 1D precipitation (ammend statistic as desired)

    # ds_precip = sf.data_catalog.get_rasterdataset(str(model_file['precip'])) # Read 2D precipitation
    # df_precip = ds_precip.to_dataframe().reset_index()[['time', 'precip']].groupby('time').mean() # Generate 1D timeseries as desired

    if 'precip' in model_file.keys():
    
        df_precip = pd.read_csv(str(model_file['precip']), index_col='time', parse_dates=['time']) # Read precip time-series
        sf.setup_precip_forcing(timeseries=df_precip)
        # sf.setup_precip_forcing(magnitude=3) # Fixed magnitude


        if 'bnd' not in model_file.keys():
            
            sf.set_config("tref", df_precip.index[0].strftime('%Y%m%d %H%M%S'))
            sf.set_config("tstart", df_precip.index[0].strftime('%Y%m%d %H%M%S'))
            sf.set_config("tstop", df_precip.index[-1].strftime('%Y%m%d %H%M%S'))

        else:
            
            pass

    else:

        pass
    

    # sf.plot_forcing()


    # # Add infiltration

    # In[15]:


    sf.setup_cn_infiltration(str(model_file['inf']), antecedent_moisture=str(model_file['ant_moist']))


    # # Setup observations

    # In[16]:


    # Create lists of x and y coordinates based on n_dis
    gdf_obs = gpd.read_file(str(model_file['obs_points']))
    gdf_obs.index = list(range(1001, 1001 + len(gdf_obs)))

    sf.setup_observation_points(locations=gdf_obs)


    # # Add weir

    # In[17]:


    # Check weir condition and execute code if weir is set to 'YES'

    weir = 'YES'
    weir1 = 'Inputs/weir.shp'
    weir2 = 'Inputs/weir_jons.shp'
    weir3 = 'Inputs/weir_cusset.shp'

    if weir == 'YES':
        gpd_weir_1 = gpd.read_file(weir1)
        gpd_weir_2 = gpd.read_file(weir2)
        gpd_weir_3 = gpd.read_file(weir3)

        #sf.setup_structures(structures=gpd_weir_1, stype='weir', dz = 25, dep = 'Inputs/bd_alti/elevtn_5.tif')
        #sf.setup_structures(structures=gpd_weir_2, stype='weir', dz = 16.7, dep = 'Inputs/bd_alti/elevtn_5.tif')
        #sf.setup_structures(structures=gpd_weir_3, stype='weir', dz = 17.3, dep = 'Inputs/bd_alti/elevtn_5.tif')

        #weir_list = [weir1, weir2]
        #for weir_fn in weir_list:
        #    weir = weir_fn
        #    gpd_weir = gpd.read_file(weir)
        #    sf.setup_structures(structures=gpd_weir, stype='weir', dz = 25, dep = 'Inputs/bd_alti/elevtn_5.tif')
        #    break


    # # Add Observation lines (Xsections)

    # In[18]:


    # Loading a LineString GeoJson clicked by user:
    sf.setup_observation_lines(locations=str(model_file['obs_lines']), merge=True)


    # # Computational parameters

    # In[19]:


    # tstart = dispd.index[0]
    # tend = dispd.index[-1]
    # tref = dispd.index[0]

    # sf.set_config("dx", 300)
    # sf.set_config("dy", 300)
    # sf.set_config("tref", tref.strftime('%Y%m%d %H%M%S'))
    # sf.set_config("tstart", tstart.strftime('%Y%m%d %H%M%S'))
    # sf.set_config("tstop", '20021130 000000')
    # sf.set_config("dtout", 10800)
    # sf.set_config("dtmax", 300)
    # sf.set_config("dtmin", 200)
    # sf.set_config("alpha", 0.5)


    # # Write

    # In[20]:


    sf.write()

    # fig, ax = sf.plot_basemap(variable="dep", bmap="sat",zoomlevel=12)
    # # Run

    # In[21]:


    # path = str('docker run -v ' + sf.root  + ':/data deltares/sfincs-cpu')
    # print(path)

    # os.system(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('path', type=str, help='Path argument')
    args = parser.parse_args()

    build_model(args.path)