import matplotlib.pyplot as plt
import seaborn as sns
import cartopy as ct
import cartopy.feature as cfeature
from utils.mahalanobis import get_pixel_array, plot_cdf, compute_cdf, compute_rsqr
import cmasher as cmr
from utils.data_info import get_storm_details
import matplotlib as mpl
import numpy as np
from utils.compute_predictions import save_predictions, interpolate_leadtimes, add_lead_zero
import pandas as pd
import palettable
from utils.plots import plot_probability_ellipses, set_plot_rc, adjust_spines, format_spines, draw_coastlines, plot_probability_ellipses_vector, plot_banana_of_uncertainty
import cartopy as ct
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import scipy.stats as sps
import os,glob,sys
import seaborn as sns
import cartopy.crs as ccrs
from utils.tcane_data_funcs import climo_to_df, read_in_TCANE, get_TCANE_distribution, make_SHASH, calc_pctiles, get_PDF_CDF, get_RI_info 
from math import floor, ceil

ELLIPSE_COLOR = {1: 'darkviolet', 2: 'goldenrod', 3: 'dodgerblue'}
NAUTICAL_MILE_TO_KM = 1.852
FIGURE_WIDTH = 32
THETA = np.linspace(0, 2 * np.pi, 1000)
#
contours = (.1, .25, .5, .75, .9,)
KM_TO_DEG = 1.0 / 111.
NMI_TO_DEG = 1.0 / 60.
#
COLOR_DEFAULT = palettable.colorbrewer.diverging.RdYlBu_9_r.mpl_colors
COLORMAP_DEFAULT = palettable.colorbrewer.diverging.RdYlBu_9_r.get_mpl_colormap()
colors = cmr.take_cmap_colors(
            "cmr.pride", len(contours), cmap_range=(0.2, 0.8), return_fmt="hex")
#
def plot_circle(ax, radius, color):
    x = radius * np.cos(THETA)
    y = radius * np.sin(THETA)
    ax.plot(x, y, '-', color=color, linewidth=4)


def plot_ellipse(ax, sigma_u, sigma_v, rho, color):
    # r = np.sqrt(-2.0 * (np.log(1.0 - 0.67))) for Pr(capture) = 67%
    r = 1.489068584398725

    x = r * sigma_u * np.cos(THETA)
    y = r * sigma_v * (rho * np.cos(THETA) + np.sqrt(1 - rho * rho) * np.sin(THETA))
    ax.plot(x, y, '-', color=color, linewidth=4)

# # `get_plot_vars_TRACK(X_out,X_in,ttype_sel)`
#
# This function just reformats some TCANE track forecast variables for easier plotting
#
# <b>Inputs</b>:
# * `X_out`: Dataframe containing TCANE output for track forecasts [Dataframe]
# * `X_in`: Dataframe containing TCANE input data [Dataframe]
# * `ttype_sel`: time type (`erly` or `late`) [str]
#
# <b>Outputs</b>: 
# * `X_p2`: Dataframe containing reformatted TCANE track forecast data [Dataframe]
# * `X_plot`: Dataframe containing reformatted TCANE track forecast data [Dataframe]

def get_plot_vars_TRACK(X_out,X_in,ttype_sel):
    X_plot = X_out.set_index(['FHOUR','TTYPE']).xs((ttype_sel),level=1)[['LONN','LATN','DATE','MU_U','MU_V','SIGMA_U','SIGMA_V','RHO','NAME']]
    #X_plot = X_plot[~X_plot.index.duplicated()]
    X_plot[['OFDX','OFDY']] = X_in.set_index(['FHOUR'])[~X_in.set_index(['FHOUR']).index.duplicated()][['OFDX','OFDY']]
    X_plot['ftime(hr)'] = X_plot.index
    X_plot = X_plot.rename(columns={'MU_U':'mu_u','MU_V':'mu_v','SIGMA_U':'sigma_u','SIGMA_V':'sigma_v','RHO':'rho','NAME':'Name'})
    fore_vecs = np.arange(12,X_plot.dropna(how='all').index.max()+1,12)
    X_p2 = X_plot.loc[fore_vecs]
    X_p2['rho'] = X_p2['rho'].astype(float)
    X_p2['ttype'] = ttype_sel
    X_plot['ttype'] = ttype_sel
    return X_p2,X_plot
## `get_plot_lims_fore(X)`
# 
# This function estimates the limits of the track forecast graphical products using the TCANE output data. 
# 
# <b>Inputs</b>:
# * `X`: Dataframe containing TCANE output for track forecasts [Dataframe]
# 
# <b>Outputs</b>:
# * `x_lims`: x-limits for TCANE track plot, in degrees longitude [array]
# * `y_lims`: y-limits for TCANE track plot, in degrees latitude [array]
# * `sigma_X`: width of uncertainty for X-component of track forecast, in degrees longitude [array]
# * `sigma_Y`: width of uncertainty for Y-component of track forecast, in degrees latitude [array]

def get_plot_lims_fore(X):
    #
    X.columns = map(lambda x: str(x).upper(), X.columns)
    #
    KM_TO_DEG = 1.0 / 111.
    Xx = X[X['FHOUR']>0]
    sigma_X = KM_TO_DEG*Xx['SIGMA_U'].astype(float).values
    sigma_Y = KM_TO_DEG*Xx['SIGMA_V'].astype(float).values
    #
    y_min = Xx["LATN"].astype(float).values + KM_TO_DEG * Xx["MU_V"].astype(float).values - sigma_Y
    y_max = Xx["LATN"].astype(float).values + KM_TO_DEG * Xx["MU_V"].astype(float).values + sigma_Y
    #
    x_min = Xx["LONN"].astype(float).values + KM_TO_DEG * Xx["MU_U"].astype(float).values - sigma_X
    x_max = Xx["LONN"].astype(float).values + KM_TO_DEG * Xx["MU_U"].astype(float).values + sigma_X
    #
    x_lims = [int(round(x_min.min()/5.0)*5.0),int(round(x_max.max()/5.0)*5.0)]
    y_lims = [int(round(y_min.min()/5.0)*5.0),int(round(y_max.max()/5.0)*5.0)]
    return x_lims,y_lims,sigma_X,sigma_Y
### `make_track_plt(ax,Xi,X_out,fore_sel,show_all=False,contours=(.1,.25,.5,.75,.9,),alpha=0.4,)`
# 
# This function plots the ellipses of uncertainty on a map for TCANE track forecasts. 
# 
# <b>Inputs</b>:
# * `ax`: handle for desired figure
# * `Xi`: Dataframe containing TCANE input for track forecasts [Dataframe]
# * `X_out`: Dataframe containing TCANE output for track forecasts [Dataframe]
# * `fore_sel`: early or late forecasts [str]
# * `show_all`: show all forecast lead times or not? [boolean; default is False]
# * `contours`: desired uncertatiny contours [array, default is (.1,.25,.5,.75,.9,)]
# * `alpha`: transparency level for contours [float, default is 0.5]

def make_track_plt(ax,Xi,X_out,fore_sel,show_all=False,contours=(.1,.25,.5,.75,.9,),alpha=0.4,):
    # Show all forecast lead times or not? By default, we show forecasts at [12,24,36,48,60,72,96,120] hours but this can be changed if desired. 
    # Show all forecast lead times or not? By default, we show forecasts at [12,24,36,48,60,72,96,120] hours but this can be changed if desired. 
    Xi.loc[Xi['LONN']==-9999.00, 'LONN'] = np.nan
    Xi.loc[Xi['LATN']==-9999.00, 'LATN'] = np.nan
    #Xi_clim.loc[Xi_clim['LONN']==-9999.00, 'LONN'] = np.nan
    #Xi_clim.loc[Xi_clim['LATN']==-9999.00, 'LATN'] = np.nan
    if show_all:
        leadtimes = np.arange(12,121,12)
        X = Xi.set_index(['ttype']).xs(fore_sel)
    else:
        leadtimes = np.arange(12,121,12)
        ind_keep = [12,24,36,48,60,72,96,120]
        Xis = Xi[Xi.reset_index().set_index(['FHOUR']).index.isin(ind_keep)]
        X = Xis.reset_index().set_index(['ttype']).xs(fore_sel)
    # Get the coordinates for the ellipses of uncertainty
    x_lims,y_lims,details,ax0 = plot_probability_ellipses(
        X,
        ax=ax,
        leadtimes=leadtimes,
        contours=contours,
        extent=None,
        alpha=0.4,
        vector=True,
        )
#
    # This is all to calculate the plot limits so that we can show all the ellipses but not have the plots be huge
    plot0_lon = float(X_out.set_index(['FHOUR','TTYPE']).loc[(0,'late')]['LONN'])
    plot0_lat = float(X_out.set_index(['FHOUR','TTYPE']).loc[(0,'late')]['LATN'])
    # Check x and y lims for outliers and replace them if they have outliers
    # check for outliers
    x_lims = [plot0_lon -10 if x < 0 else x for x in x_lims]
    y_lims = [plot0_lat - 10 if x < 0 else x for x in y_lims]
    # positive outliers
    x_lims = [plot0_lon + 20 if x > 360 else x for x in x_lims]
    y_lims = [plot0_lat + 10 if x > 90 else x for x in y_lims]
    #
    x_spread = round((max(x_lims) - min(x_lims))/5)*5
    y_spread = round((max(y_lims) - min(y_lims))/5)*5
    x_margin = x_spread/4
    y_margin = y_spread/4
    #
    x_spread2 = round((max(Xi['LONN']) - min(Xi['LONN']))/5)*5
    y_spread2 = round((max(Xi['LATN']) - min(Xi['LATN']))/5)*5
    x_margin2 = x_spread2/4
    y_margin2 = y_spread2/4
    #
    x_ext_min = min(min(x_lims),Xi['LONN'].min())
    x_ext_max = max(max(x_lims),Xi['LONN'].max())
    y_ext_min = min(min(y_lims),Xi['LATN'].min())
    y_ext_max = max(max(y_lims),Xi['LATN'].max())
    #
    #
    ax.plot(plot0_lon,plot0_lat,'x',color='k',markersize=7,transform=ct.crs.PlateCarree(central_longitude=0.))
    ax.text(plot0_lon+0.5,plot0_lat+0.25,'0',fontsize=10,transform=ct.crs.PlateCarree(central_longitude=0.))
    #ax.set_extent([-95,-72,15,37.5])
    # ax.set_extent([X['LONN'].min()-11,X['LONN'].max()+5,X['LATN'].min()-5,X['LATN'].max()+5])
    #ax.set_title('{name}, {date}, {fore_sel}'.format(name=X['Name'].iloc[0],date=X['DATE'].iloc[0],
                                                 #fore_sel=X['fore sel'],fontsize=40))
    #
    # xdiff = max(xlims) - min(xlims)
    # ydiff = max(ylims) - min(ylims)
    # print(x_ext_min,x_ext_max)
    plotx_ll = max(x_ext_min-x_margin,220)
    plotx_up = min(x_ext_max+x_margin,359)
    ploty_ll = max(y_ext_min-y_margin,5)
    ploty_up = y_ext_max+y_margin
    #if plot
    # print(plotx_ll,plotx_up,ploty_ll,ploty_up)
    ax.set_extent([0.975*plotx_ll,
               1.025*plotx_up,
               ploty_ll,
               ploty_up],
             crs=ccrs.PlateCarree(central_longitude=0.))
    #ax.set_extent([220,300,10,45],crs=ccrs.PlateCarree(central_longitude=0.))
    if fore_sel == 'erly':
        ttype_plt = 'early'
    else:
        ttype_plt = fore_sel
    ax.set_title('{name}, {date}, {fore_sel}'.format(name=Xi['Name'].iloc[0],date=Xi['DATE'].iloc[0],fore_sel=ttype_plt),fontsize=22)
    return ax

## `make_track_plt_climo(ax,Xi,X_out,Xi_clim,fore_sel,cmax=np.round(2/3,3),show_all=False,contours=(.1,.25,.5,.75,.9,),alpha=0.4,use_gradient_color=True)`
# 
# This function plots the ellipses of uncertainty specified by `cmax` for TCANE track forecasts and TCANE climo track forecasts. 
# 
# <b>Inputs</b>:
# * `ax`: handle for desired figure
# * `Xi`: Dataframe containing TCANE input for track forecasts [Dataframe]
# * `X_out`: Dataframe containing TCANE output for track forecasts [Dataframe]
# * `Xi_clim`: Dataframe containing TCANE climatology output for track forecasts [Dataframe]
# * `fore_sel`: early or late forecasts [str]
# * `cmax`: desired uncertatiny contour [float, default is 2/3]
# * `show_all`: show all forecast lead times or not? [boolean; default is False]
# * `contours`: desired uncertatiny contours [array, default is (.1,.25,.5,.75,.9,)]
# * `alpha`: transparency level for contours [float, default is 0.5]
# * `use_gradient_color`: do we want the contours to have a color gradient or all be one color? [boolean, default is false]
def make_track_plt_climo(ax,Xi,X_out,Xi_clim,fore_sel,cmax=np.round(2/3,3),show_all=False,alpha=0.4,use_gradient_color=True):
    # Show all forecast lead times or not? By default, we show forecasts at [12,24,36,48,60,72,96,120] hours but this can be changed if desired. 
    Xi.loc[Xi['LONN']==-9999.00, 'LONN'] = np.nan
    Xi.loc[Xi['LATN']==-9999.00, 'LATN'] = np.nan
    Xi_clim.loc[Xi_clim['LONN']==-9999.00, 'LONN'] = np.nan
    Xi_clim.loc[Xi_clim['LATN']==-9999.00, 'LATN'] = np.nan
    if show_all:
        leadtimes = np.arange(12,121,12)
        X = Xi.set_index(['ttype']).xs(fore_sel)
        X_clim = Xi_clim.set_index(['ttype']).xs(fore_sel)
    else:
        leadtimes = np.arange(12,121,12)
        ind_keep = [12,24,36,48,60,72,96,120]
        Xis = Xi[Xi.reset_index().set_index(['FHOUR']).index.isin(ind_keep)]
        Xic = Xi_clim[Xi_clim.reset_index().set_index(['FHOUR']).index.isin(ind_keep)]
        X = Xis.reset_index().set_index(['ttype']).xs(fore_sel)
        X_clim = Xic.reset_index().set_index(['ttype']).xs(fore_sel)
   # Plot climo ellipse
    x_lims,y_lims,details,ax0 = plot_probability_ellipses(
        X_clim,
        ax=ax,
        leadtimes=leadtimes,
        contours=(cmax,),
        extent=None,
        alpha=alpha,
        annotate_leadtimes=True,
        colors=None,
        # colors = None,
        vector=True,
        plot_nhc_cone=False,
        is_fill=False,
        linetype='--',
        linewt=2,
        no_label=True,
        is_climo=True,
        use_gradient_color=True)
    # Plot TCANE forecast ellipse
    x_lims2,y_lims2,details2,ax2 = plot_probability_ellipses(
        X,
        ax=ax,
        leadtimes=leadtimes,
        contours=(cmax,),
        extent=None,
        alpha=1-alpha,
        annotate_leadtimes=True,
        colors=None,
       # colors = None,
        vector=True,
        plot_nhc_cone=False,
        is_fill=False,
        linetype='-',
        linewt=4,
        use_gradient_color=True)
# This is all to calculate the plot limits so that we can show all the ellipses but not have the plots be huge
    plot0_lon = float(X_out.set_index(['FHOUR','TTYPE']).loc[(0,'late')]['LONN'])
    plot0_lat = float(X_out.set_index(['FHOUR','TTYPE']).loc[(0,'late')]['LATN'])
    # Check x and y lims for outliers and replace them if they have outliers
    # check for outliers
    x_lims = [plot0_lon -10 if x < 0 else x for x in x_lims]
    x_lims2 = [plot0_lon - 10 if x < 0 else x for x in x_lims2]
    y_lims = [plot0_lat - 10 if x < 0 else x for x in y_lims]
    y_lims2 = [plot0_lat - 10 if x < 0 else x for x in y_lims2]
    # positive outliers
    x_lims = [plot0_lon + 20 if x > 360 else x for x in x_lims]
    x_lims2 = [plot0_lon + 20 if x > 360 else x for x in x_lims2]
    y_lims = [plot0_lat + 10 if x > 90 else x for x in y_lims]
    y_lims2 = [plot0_lat + 10 if x > 90 else x for x in y_lims2]
    #
    ax.plot(plot0_lon,plot0_lat,'x',color='k',markersize=10,transform=ct.crs.PlateCarree(central_longitude=0.))
    ax.text(plot0_lon+0.5,plot0_lat+0.25,'0',fontsize=12,transform=ct.crs.PlateCarree(central_longitude=0.))
    # 
    x_spread = round((max(x_lims) - min(x_lims))/5)*5
    y_spread = round((max(y_lims) - min(y_lims))/5)*5
    x_margin = x_spread/4
    y_margin = y_spread/4
    #
    x_spread2 = round((max(x_lims2) - min(x_lims2))/5)*5
    y_spread2 = round((max(y_lims2) - min(y_lims2))/5)*5
    x_margin2 = x_spread2/4
    y_margin2 = y_spread2/4
    # x_ext_min = min(min(min(x_lims)-x_margin,min(x_lims2)-x_margin2),0)
   # x_ext_max = max(max(x_lims)+x_margin,max(x_lims2)+x_margin2)
    # y_ext_min = min(min(y_lims)-y_margin,min(y_lims2)-y_margin2)
    # y_ext_max = max(max(y_lims)+y_margin,max(y_lims2)+y_margin2)
    x_ext_min = min(x_lims,x_lims2)
    x_ext_max = max(x_lims,x_lims2)
    y_ext_min = min(y_lims,y_lims2)
    y_ext_max = max(y_lims,y_lims2)
    # print('x extent is [{xm}, {xmx}]'.format(xm=x_ext_min,xmx=x_ext_max))
    # print('y extent is [{ym}, {ymx}]'.format(ym=y_ext_min,ymx=y_ext_max))
    # print('x_margin is {xmar}, y_margin is {ymar}'.format(xmar=x_margin,ymar=y_margin))
    plotx_ll = max(x_ext_min)-x_margin
    plotx_up = min(max(x_ext_max)+x_margin/2,359)
    ploty_ll = min(y_ext_min)-y_margin
    ploty_up = max(y_ext_max)+y_margin
    #if plot
    ax.set_extent([0.975*min(min(x_lims),min(x_lims2)),
               1.025*max(max(x_lims),max(x_lims2)),
               ploty_ll,
               ploty_up],
             crs=ccrs.PlateCarree(central_longitude=0.))
    #
    if fore_sel == 'erly':
        ttype_plt = 'early'
    else:
        ttype_plt = fore_sel
    ax.set_title('{name}, {date}, {fore_sel}'.format(name=Xi['Name'].iloc[0],date=Xi['DATE'].iloc[0],fore_sel=ttype_plt),fontsize=22)
    return ax
