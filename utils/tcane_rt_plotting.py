import numpy as np
import pandas as pd
import os,glob,re,sys
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import palettable
from scipy.stats import skewnorm, rv_histogram
import scipy.stats as sps
import cartopy as ct
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import os,glob,sys
import cartopy.crs as ccrs
from utils.tcane_data_funcs import climo_to_df, read_in_TCANE, get_TCANE_distribution, make_SHASH, calc_pctiles, get_PDF_CDF, get_RI_info 
from utils.tcane_plotting import make_boxplot, get_cat_probs, plot_RI, make_pctile_plot, make_all_plotting_data, plot_RI_bar, make_TCANE_dists_pdf_cdf
from utils.bdeck_edeck_funcs import get_bdecks, get_edeck_probs
from utils.tcane_track_plotting import plot_circle, plot_ellipse, get_plot_vars_TRACK, get_plot_lims_fore, make_track_plt, make_track_plt_climo 
from utils.plots import plot_probability_ellipses, set_plot_rc, format_spines, adjust_spines, draw_coastlines, plot_probability_ellipses_vector, plot_banana_of_uncertainty 
from utils.mahalanobis import get_pixel_array, plot_cdf, compute_cdf, compute_rsqr
# from plots import plot_probability_ellipses

# 1. Call functions to make intensity forecast plots
# This function plots the TCANE intensity forecast and compares it to the TCANE consensus forecast (for early forecasts) or the official NHC forecast (for late forecasts). 
#
# <b>Inputs</b>:
# * `pct_e`: Dataframe containing the TCANE intensity forecast uncertainty distribution [dataframe]
# * `dfout`: TCANE output data file [dataframe]
# * `dfin`: TCANE input data file [dataframe]
# * `stormdate`: contains storm ID and forecast ID, used to label and save plots [str]
# * `targetdir`: output directory where figures will be saved [str]
# * `type_sel`: indicates early or late forecasts [`erly` or `late`, str]

def TCANE_plots_intensity_forecasts(pct_e,dfout,dfin,stormdate,targetdir,type_sel):
    # Make plot
    if type_sel == 'erly':
        type_sel_plt = 'EARLY'
    else:
        type_sel_plt = type_sel.upper()
    fig1,ax1a = plt.subplots(1,1,figsize=(12,8))
    pct_all = pd.concat([pct_e])
    ax1a = make_pctile_plot(ax1a,pct_all,type_sel,dfout,dfin)
    # ax1b = make_pctile_plot(ax1b,pct_all,'late',dfout,dfin)
    #
    fig1.suptitle('TCANE Forecasts, {name}, {ex_date} ({type_sel})'.format(name=dfin.iloc[0]['NAME'],ex_date=stormdate,type_sel=type_sel_plt),fontsize=36,y=1.02)
    fig1.tight_layout()
    fig1.savefig('{target_savedir}/p1-99_{exdate}_{type_sel}.pdf'.format(target_savedir=targetdir,exdate=stormdate,type_sel=type_sel_plt),format='pdf',bbox_inches='tight')
    fig1.savefig('{target_savedir}/p1-99_{exdate}_{type_sel}.png'.format(target_savedir=targetdir,exdate=stormdate,type_sel=type_sel_plt),format='png',dpi=400,bbox_inches='tight')
    ## Same plot as above but with best tracks
    # fig2,(ax2a,ax2b) = plt.subplots(1,2,figsize=(20,9))
    # pct_all = pd.concat([pct_e,pct_l])
    # ax2a = make_pctile_plot(ax2a,pct_all,'erly',dfout,dfin,bdeck_trim)
    # ax2b = make_pctile_plot(ax2b,pct_all,'late',dfout,dfin,bdeck_trim)
    #
    # fig2.suptitle('TCANE Forecasts, {name}, {ex_date}'.format(name=dfin.iloc[0]['NAME'],ex_date=stormdate),fontsize=36,y=1.02)
    # fig2.tight_layout()
    # fig2.savefig('{target_savedir}/p1-99_{name}_{exdate}_with_BTR.pdf'.format(target_savedir=targetdir,name=dfin.iloc[0]['NAME'],exdate=stormdate),format='pdf',bbox_inches='tight')
    # fig2.savefig('{target_savedir}/p1-99_{name}_{exdate}_with_BTR.png'.format(target_savedir=targetdir,name=dfin.iloc[0]['NAME'],exdate=stormdate),format='png',dpi=400,bbox_inches='tight')
    return ('intensity forecats plots done')

# 2. Call functions to make category and RI plots
# This function plots the TCANE RI forecast and category probability plots. 
#
# <b>Inputs</b>:
# * `TC_e`: Dataframe containing the TCANE intensity forecast Category 1...5 probabilities [dataframe]
# * `c_TC_e`: Dataframe containing the intensity forecast Category 1...5 probabilities for TCANE climo [dataframe]
# * `df_in`: TCANE input data file [dataframe]
# * `storm_date`: contains storm ID and forecast ID, used to label and save plots [str]
# * `target_savedir`: output directory where figures will be saved [str]
# * `type_sel`: indicates early or late forecasts [`erly` or `late`, str]
# * `RI_e`: Dataframe containing TCANE rapid intensification forecasts [dataframe]
# * `edeck_all`: Dataframe containing edeck intensity forecasts for given storm [dataframe]
# * `bas_ab`: Abbreviation for forecast basin [str]
# * `c_RI_e`: Dataframe containing rapid intensification forecasts for TCANE climo [dataframe]

def TCANE_plots_cat_and_RI_plots(TC_e,c_TC_e,df_in,storm_date,target_savedir,type_sel,RI_e,edeck_all,bas_ab,c_RI_e):
    if type_sel == 'erly':
        type_sel_plt = 'EARLY'
    else:
        type_sel_plt = type_sel.upper()
    fig20,ax20 = plt.subplots(1,1,figsize=(10,6))
    ax20 = get_cat_probs(ax20,TC_e,c_TC_e,df_in,type_sel)
    # ax20b = get_cat_probs(ax20b,TC_l,c_TC_l,df_in,'late')
    fig20.suptitle('{name}, {exdate} ({type_sel})'.format(name=df_in.iloc[0]['NAME'],exdate=storm_date,type_sel=type_sel_plt),fontsize=35,y=1.025)
    fig20.tight_layout()
    fig20.savefig('{target_savedir}/pr_cat_{exdate}_{type_sel}.pdf'.format(target_savedir=target_savedir,exdate=storm_date,type_sel=type_sel_plt),format='pdf',bbox_inches='tight')
    fig20.savefig('{target_savedir}/pr_cat_{exdate}_{type_sel}.png'.format(target_savedir=target_savedir,exdate=storm_date,type_sel=type_sel_plt),format='png',dpi=400,bbox_inches='tight')
    # Probability of RI
    fig22,ax22 = plt.subplots(1,1,figsize=(10,6))
    RI_all = RI_e
    ax22 = plot_RI(ax22,RI_all,edeck_all)
    ax22.set_title('Prob. of RI for {name} ({exdate}, {type_sel})'.format(name=df_in.iloc[0]['NAME'],exdate=storm_date,type_sel=type_sel_plt),fontsize=28)
    fig22.tight_layout()
    fig22.savefig('{target_savedir}/pr_RI_{exdate}_{type_sel}_with_edeck.pdf'.format(target_savedir=target_savedir,exdate=storm_date,type_sel=type_sel_plt),format='pdf',bbox_inches='tight')
    fig22.savefig('{target_savedir}/pr_RI_{exdate}_{type_sel}_with_edeck.png'.format(target_savedir=target_savedir,exdate=storm_date,type_sel=type_sel_plt),format='png',dpi=400,bbox_inches='tight')
    # Same as above but without the edeck models
    fig22,ax22 = plt.subplots(1,1,figsize=(10,6))
    # RI_all = pd.concat([RI_e,RI_l])
    ax22 = plot_RI(ax22,RI_all)#,edeck_all)
    ax22.set_title('Prob. of RI for {name} ({exdate}, {type_sel})'.format(name=df_in.iloc[0]['NAME'],exdate=storm_date,type_sel=type_sel_plt),fontsize=28)
    fig22.tight_layout()
    fig22.savefig('{target_savedir}/pr_RI_{exdate}_{type_sel}.pdf'.format(target_savedir=target_savedir,exdate=storm_date,type_sel=type_sel_plt),format='pdf',bbox_inches='tight')
    fig22.savefig('{target_savedir}/pr_RI_{exdate}_{type_sel}.png'.format(target_savedir=target_savedir,exdate=storm_date,type_sel=type_sel_plt),format='png',dpi=400,bbox_inches='tight')
    ## 
    fig77,ax77e = plt.subplots(1,1,figsize=(10,6))
    plot_RI_bar(ax77e,RI_all,type_sel,c_RI_e,bas_ab,edeck_all=pd.DataFrame())
    ax77e.set_title('Prob. of RI for {name} ({exdate}, {type_sel})'.format(name=df_in.iloc[0]['NAME'],exdate=storm_date,type_sel=type_sel_plt),fontsize=24)
    fig77.tight_layout()
    fig77.savefig('{target_savedir}/barplot_pr_RI_{exdate}_{type_sel}.pdf'.format(target_savedir=target_savedir,exdate=storm_date,type_sel=type_sel_plt),format='pdf',bbox_inches='tight')
    fig77.savefig('{target_savedir}/barplot_pr_RI_{exdate}_{type_sel}.png'.format(target_savedir=target_savedir,exdate=storm_date,type_sel=type_sel_plt),format='png',dpi=400,bbox_inches='tight')
    #
    
    return ('RI and Pr(Category) plots done')

# #3. Make track probability plots. Default chosen contour is 66.667% but user can change `cmax` argument if they want to plot a different contour. 

# This function plots the TCANE track forecast uncertainty at a level specified by `cmax` and compares this TCANE forecast uncertainty to a TCANE climatological forecast uncertainty. 
#
# <b>Inputs</b>:
# * `track_sub`: Dataframe containing the TCANE track forecast variables [dataframe]
# * `df_out`: TCANE output data file [dataframe]
# * `track_sub_clim`: Dataframe containing the TCANE track forecast variables for TCANE_climo [dataframe]
# * `storm_date`: contains storm ID and forecast ID, used to label and save plots [str]
# * `df_in`: TCANE input data file [dataframe]
# * `target_savedir`: output directory where figures will be saved [str]
# * `type_sel`: indicates early or late forecasts [`erly` or `late`, str]
# * `cmax`: specified uncertainty level for TCANE plots; must be a fraction between 0 and 1. Default is 0.6667 [float]

def TCANE_track_plots_with_climo(track_sub,df_out,track_sub_clim,storm_date,df_in,target_savedir,type_sel,cmax=np.round(2/3,3)):
    if type_sel == 'erly':
        type_sel_plt = 'EARLY'
    else:
        type_sel_plt = type_sel.upper()
    fig5 = plt.figure(figsize=(15,12))
    ax5 = fig5.add_subplot(1,1,1,projection=ct.crs.PlateCarree(central_longitude=0.))
    ax5 = make_track_plt_climo(ax5,track_sub,df_out,track_sub_clim,type_sel,cmax=cmax,show_all=False)
    # ax5.set_ylim([mn.values[0],mx.values[0]])
    #ax5b = fig5.add_subplot(1,2,2,projection=ct.crs.PlateCarree(central_longitude=0.))
    # ax5b = make_track_plt_climo(ax5b,track_sub,df_out,track_sub_clim,'late',cmax=cmax,show_all=False)
    # ax5b.set_ylim([mn.values[0],mx.values[0]])
    fig5.tight_layout()
    # fig5.suptitle('{name}, {date}'.format(name=track_sub['Name'].iloc[0],date=track_sub['DATE'].iloc[0],
         #                                  fontsize=75),y=1.01)
    fig5.savefig('{target_savedir}/TRACK_climo_{cmax}_pctile_{exdate}_{type_sel}.pdf'.format(cmax=np.round(cmax*100,0).astype(int),
            target_savedir=target_savedir,exdate=storm_date,type_sel=type_sel_plt),format='pdf',bbox_inches='tight')
    fig5.savefig('{target_savedir}/TRACK_climo_{cmax}_pctile_{exdate}_{type_sel}.png'.format(cmax=np.round(cmax*100,0).astype(int),
            target_savedir=target_savedir,exdate=storm_date,type_sel=type_sel_plt),format='png',dpi=400,bbox_inches='tight')
    return ('Track plotted with climo for ',cmax)

# #4. Make track plot all ellipses

# This function makes the TCANE track plot with the full uncertainty distribution. 
#
# <b>Inputs</b>:
# * `track_sub`: Dataframe containing the TCANE track forecast variables [dataframe]
# * `df_out`: TCANE output data file [dataframe]
# * `target_savedir`: output directory where figures will be saved [str]
# * `type_sel`: indicates early or late forecasts [`erly` or `late`, str]
# * `df_in`: TCANE input data file [dataframe]
# * `storm_date`: contains storm ID and forecast ID, used to label and save plots [str]

def TCANE_track_plot_all(track_sub,df_out,target_savedir,type_sel,df_in,storm_date):
    if type_sel == 'erly':
        type_sel_plt = 'EARLY'
    else:
        type_sel_plt = type_sel.upper()
    fig4 = plt.figure(figsize=(12,8))
    ax4 = fig4.add_subplot(1,1,1,projection=ct.crs.PlateCarree(central_longitude=0.))
    # Drop forecast times with missing data
    #track_sub = track_sub[track_sub.LATN != -9999.00]
    track_sub = track_sub[track_sub.LONN != -9999.00]
    track_sub = track_sub.dropna(subset = ['LONN','LATN'])
    # print(track_sub)
    ax4 = make_track_plt(ax4,track_sub,df_out,type_sel,show_all=False)
    mx = df_out[['LATN']].astype(float).max()*1.5
    mn = df_out[['LATN']].astype(float).min()*0.75
    # ax4.set_ylim([mn.values[0],mx.values[0]])
    resize_x = ax4.get_xlim()
    resize_y = ax4.get_ylim()
    # print(resize_x,resize_y)
    if max(resize_x) - min(resize_x) > 50:
        ax4.set_extent([resize_x[0]*.75,resize_x[1]*1.15,
                        resize_y[0]*0.95,resize_y[1]*.85])
    # ax4.set_extent([resize_x[0],resize_x[1],resize_y[0],resize_y[1]])
    fig4.tight_layout()
    #fig4.suptitle('{name}, {date}'.format(name=track_sub['Name'].iloc[0],date=track_sub['DATE'].iloc[0],
    #                                      fontsize=75),y=1.01)
    fig4.savefig('{target_savedir}/TRACK_{exdate}_{type_sel}.pdf'.format(target_savedir=target_savedir,
       exdate=storm_date,type_sel=type_sel_plt),format='pdf',bbox_inches='tight')
    fig4.savefig('{target_savedir}/TRACK_{exdate}_{type_sel}.png'.format(target_savedir=target_savedir,
        exdate=storm_date,type_sel=type_sel_plt),format='png',dpi=400,bbox_inches='tight')
    return ('Track forecasts with all probability ellipses plotted')

# #5. ### `tcane_plotting_make_ALL`

# This function puts it all together and reads in the relevant files for the forecast specified by `storm_ID` and `forecast_ID`, calculates the TCANE forecast distribution, and makes the desired graphics. The TCANE graphics will be located in `Figures/storm_ID/`, and are saved in both PDF and PNG format. 
# 
# <b>Inputs</b>:
# * `in_dir`: Directory where the TCANE <b>input</b> files are located. This will change based on whichever machine you are running on, but should not change from one forecast to the next [str]
# * `out_dir`:Directory where the TCANE <b>output</b> files are located. This will change based on whichever machine you are running on, but should not change from one forecast to the next [str]
# * `clim_dir`: Directory where the TCANE <b>climatology</b> files are located. This will change based on whichever machine you are running on, but should not change from one forecast to the next [str]
# * `bdeck_dir`: Directory where the real-time ATCF b-deck and e-deck files rae located. This will change based on whichever machine you are running on, but should not change from one forecast to the next [str]
# * `storm_ID`: 8-character ATCF ID for the storm of interest. Should have the form of BBNNYYYY, where `BB` is the two-character storm basin, `NN` is the two character storm number (single digit numbers will begin with a 0, e.g., the 2nd storm of the year will be '02'), and `YYYY` is the four-character year. [str]
# * `forecast_ID`: 6-character forecast ID for the storm of interest. Should have the form of MMDDHH, where `MM` is month, `DD` is day, and `HH` is hour. [str]
# 
# <b>Outputs</b>:
# A print statement indicating that the graphics have been made for `storm_ID`_`forecast_ID`
def tcane_plotting_make_ALL(in_dir,out_dir,clim_dir,bdeck_dir,storm_ID,forecast_ID):
    # Make sure that ATCFID is in capital letters
    storm_ID = storm_ID.upper()
    print(storm_ID)
    # Create output directory for figures
    if not os.path.isdir('Figures/'):
        os.mkdir('Figures/')
    target_savedir = 'Figures/{storm}/'.format(storm=storm_ID)
    if not os.path.isdir(target_savedir):
        os.mkdir(target_savedir)
    else:
        print('directory exists')
    # Get basin from ATCFID
    bas_ab = storm_ID[0:2].lower()
    # Locate all files associated with that storm and forecast
    fnames_input =  glob.glob(in_dir+'{storm_ID}_{forecast}*input.dat'.format(storm_ID=storm_ID,forecast=forecast_ID))
    fnames_output = glob.glob(out_dir+'{storm_ID}_{forecast}*output.dat'.format(storm_ID=storm_ID,forecast=forecast_ID))
    # Get full filename
    fi = fnames_input[0]
    storm_date = storm_ID+'_'+forecast_ID[4:]
    print('running case ',storm_date)
    fo = glob.glob(out_dir+'{storm_date}_{forecast}*output.dat'.format(storm_date=storm_ID,forecast=forecast_ID))[0]
    # Load TCANE datasets
    df_climo = climo_to_df(clim_dir+'tcane_climo_format_{ba}.dat'.format(ba=bas_ab))
    df_in = read_in_TCANE(fi)
    df_in.replace('-9999.00',np.nan,inplace=True)
    df_in.replace('-9999.0',np.nan,inplace=True)
    df_out = read_in_TCANE(fo)
    df_out.replace('-9999.00',np.nan,inplace=True)
    df_out.replace('-9999.0',np.nan,inplace=True)
    # Make sure that forecast times match
    df_in = df_in[df_in['FHOUR'].isin(df_out['FHOUR'])]
    # Get realtime b-deck and e-deck files that correspond to storm_ID
    b_deck_ALL,b_deck_trim = get_bdecks(storm_ID[4:8],storm_ID[2:4],bas_ab,bdeck_dir)
    edeck_all = get_edeck_probs(bdeck_dir,storm_date,bas_ab)
    # Check if late forecasts are available or not. If not, we will only make plots for early forecasts
    if all(df_out.loc[df_out['TTYPE']=='late', 'VMXC'] == -9999.0):
        print('no late forecasts available')
        skip_late = True
    else:
        print('make early and late forecasts please')
        skip_late = False
    # Run early code no matter what. This should always be available 
    # Get TCANE distributions and PDF/CDFs
    tcane_dist_ERLY,Yshash_erly,pdf_cdf_erly = make_TCANE_dists_pdf_cdf(df_out,df_in,'erly')
    vmax = float(df_out['VMAXN'].max())
    pvc = [.01,.05,.1,.25,.5,.75,.9,.95,.99]
    # Get relevant data for intensity plots--intensity distributions, RI plots, and so on
    Y_e,pdf_e,pct_e,RI_e,TC_e = make_all_plotting_data(df_in,df_out,vmax,'erly',pvc)
    # Get relevant data for intensity plots using TCANE Climo
    c_Y_e,c_pdf_e,c_pct_e,c_RI_e,c_TC_e = make_all_plotting_data(df_in,df_climo,vmax,'erly',pvc)
    # Intensity forecast plots
    TCANE_plots_intensity_forecasts(pct_e,df_out,df_in,storm_date,target_savedir,'erly')
    # Plots of category probability and RI probability
    TCANE_plots_cat_and_RI_plots(TC_e,c_TC_e,df_in,storm_date,target_savedir,'erly',RI_e,edeck_all,bas_ab,c_RI_e)
    # 
    # Now, we get plotting variables for track
    df_climo[['ATCFID','NAME']] = 'Climo'
    df_climo[['DATE','LATN','LONN']] = df_out[['DATE','LATN','LONN']]
    # Track variables for early forecasts 
    track_sub2_e,track_xplot_e = get_plot_vars_TRACK(df_out,df_in,ttype_sel='erly')
    # Track variables using TCANE climo for climo plots 
    track_sub_clim_e,track_xplot_clim_e = get_plot_vars_TRACK(df_climo,df_in,ttype_sel='erly')
    track_sub2_e[['LONN','LATN','mu_u','mu_v','sigma_u','sigma_v','rho','OFDX','OFDY']] = track_sub2_e[['LONN','LATN','mu_u','mu_v','sigma_u','sigma_v','rho','OFDX','OFDY']].astype(float)
    track_sub_clim_e[['LONN','LATN','mu_u','mu_v','sigma_u','sigma_v','rho','OFDX','OFDY']] = track_sub_clim_e[['LONN','LATN','mu_u','mu_v','sigma_u','sigma_v','rho','OFDX','OFDY']].astype(float)
    # Check for end of forecast:
    if len(track_sub2_e['ftime(hr)'].unique()) < 2:
        print('end of the forecast time') 
    else:
        # Make TCANE track plots. 
        TCANE_track_plots_with_climo(track_sub2_e,df_out,track_sub_clim_e,storm_date,df_in,target_savedir,'erly')
        TCANE_track_plots_with_climo(track_sub2_e,df_out,track_sub_clim_e,storm_date,df_in,target_savedir,'erly',cmax=0.95)
        TCANE_track_plot_all(track_sub2_e,df_out,target_savedir,'erly',df_in,storm_date)
    # Now, if the late forecasts are available, run code for the late forecasts
    plt.close('all')
    if not skip_late:
        df_out = df_out.replace('-9999.0',np.nan)
        # Get TCANE plot data and RI information for late forecasts
        Y_l,pdf_l,pct_l,RI_l,TC_l = make_all_plotting_data(df_in,df_out,vmax,'late',pvc)
        # Get TCANE plot data and RI information for late forecasts using climatology for climo plots
        c_Y_l,c_pdf_l,c_pct_l,c_RI_l,c_TC_l = make_all_plotting_data(df_in,df_climo,vmax,'late',pvc)
        # Make intensity forecast plots for late forecasts 
        TCANE_plots_intensity_forecasts(pct_l,df_out,df_in,storm_date,target_savedir,'late')
        # Plots of category probability and RI probability
        TCANE_plots_cat_and_RI_plots(TC_l,c_TC_l,df_in,storm_date,target_savedir,'late',RI_l,edeck_all,bas_ab,c_RI_l)
        # Track plot variables for late forecasts
        track_sub2_l,track_xplot_l = get_plot_vars_TRACK(df_out,df_in,ttype_sel='late')
        track_sub_clim_l,track_xplot_clim_l = get_plot_vars_TRACK(df_climo,df_in,ttype_sel='late')
        track_sub2_l[['LONN','LATN','mu_u','mu_v','sigma_u','sigma_v','rho','OFDX','OFDY']] = track_sub2_l[['LONN','LATN','mu_u','mu_v','sigma_u','sigma_v','rho','OFDX','OFDY']].astype(float)
        track_sub_clim_l[['LONN','LATN','mu_u','mu_v','sigma_u','sigma_v','rho','OFDX','OFDY']] = track_sub_clim_l[['LONN','LATN','mu_u','mu_v','sigma_u','sigma_v','rho','OFDX','OFDY']].astype(float)
        # Check for end of forecast:
        if len(track_sub2_l['ftime(hr)'].unique()) < 2:
            print('end of the forecast time') 
        else:
            # Actually make track plots for late forecasts
            TCANE_track_plots_with_climo(track_sub2_l,df_out,track_sub_clim_l,storm_date,df_in,target_savedir,'late')
            TCANE_track_plots_with_climo(track_sub2_l,df_out,track_sub_clim_l,storm_date,df_in,target_savedir,'late',cmax=0.95)
            TCANE_track_plot_all(track_sub2_l,df_out,target_savedir,'late',df_in,storm_date)
    else:
        print('Late forecasts not available yet')
    plt.close('all')
    return "done running for ",storm_ID,"_",forecast_ID
