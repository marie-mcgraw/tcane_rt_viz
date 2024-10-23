# tcane_rt_viz
This code makes real-time graphical output for the TCANE model. After you have cloned this repo using `git clone`, navigate to the correct directory (`tcane_realtime_plotting`). 

1. <b>Set up the environment file</b> using `conda`: `conda env create -f environment.yml` (note that creating the environment might take a few minutes; don't panic! You only have to do this part once). This environment can only be used to run the realtime plotting code–we can’t train the model in this environment.

2. To <b>activate the environment</b>, run `conda activate tcane_rt`. To deactivate, the command is `conda deactivate`. The Python environment will need to be activated before running the plotting code. It should not affect any Fortran operations. You will need to activate the environment for every session where you want to run the visualization code. To activate an environment in a shell script, use the command `source activate tcane_rt`. You can deactivate the Python environment at the end of the script using `deactivate tcane_rt`.  

3. Make sure your <b>directory structure</b> is correct: You should have a main directory that contains a Python script (`run_TCANE_rt_plotting.py`), a Jupyter notebook (`run_TCANE_rt_plotting.ipynb`), the environment file (`environment.yml`), and two subdirectories, `utils` and `Figures`. `utils` contains all of the functions and subroutines used to make the TCANE visuals. A more detailed description of what each script in `utils` contains is at the end of this readme. If the `Figures` directory doesn’t exist yet, don’t worry, the plotting code will create it for you.

4.  <b>Running the TCANE visualization code</b>: The plotting code needs realtime TCANE input files and output files; TCANE climatology; and real-time b-deck and e-deck files to run. First, the plotting code needs to know where the TCANE files are located. The user needs to specify the relevant directories for TCANE input, output, and climatology files; we also need to specify the relevant directories for the real-time b-deck and e-deck files. These directories are specified within the Python script (`run_TCANE_rt_plotting.py`), as we expect these file paths to stay the same from forecast to forecast. <b>The user should go into this script and update the directories before running this code for the first time</b>; afterwards, these directories should not need to be changed. The directories are: 

  * `out_dir` is the directory where the TCANE output files are located. (`out_dir`, line 44 of `run_TCANE_rt_plotting.py`)
  * `clim_dir` is the directory where the TCANE climatology files are located. (`clim_dir`, line 45 of `run_TCANE_rt_plotting.py`)
  * `in_dir` is the directory where the TCANE input files are located. (`in_dir`, line 46 of `run_TCANE_rt_plotting.py`)
  * `bdeck_dir` is the directory where realtime b-deck and e-deck files are located. (`bdeck_dir`, line 47 of `run_TCANE_rt_plotting.py`)

Then, the <b>visualization code can be run from the command line</b> with the following inputs: `python run_TCANE_rt_plotting.py STORMID FOREDATE`
  * `STORMID` contains the ATCFID of the desired storm, in the form `BBNNYYYY`, where `BB` is the 2-letter basin abbreviation, `NN` is the 2 character storm number, and `YYYY` is the 4 character year. [string]
  * `FOREDATE` contains the forecast date of the desired storm in the form `MMDDHH`. [string]

5.  <b>Output</b>: The main output of this code is figures. The figures will be located in `Figures/{storm_ID}/`. If this directory does not already exist, the code will create it. All figures for a given `storm_ID` will be located in the same directory. Figures will be saved in both PDF and PNG format. Separate figures for both early and late forecasts will be generated.

6.  <i>NOTE ON LATE FORECASTS</i>: If the late runs for a given forecast time are not available when the code is run, it will only produce plots for the early forecasts. 

## Table of Contents for `utils/`

The `utils/` directory contains all of the essential code to create the TCANE visualizations. This code should only be of interest to the TCANE development team, though end users may want to make small changes such as changing color schemes, updating default settings, and so on. The following files are contained in `utils/`:

`tcane_rt_plotting.py`: This file is where we call most of the plotting code. It contains three functions:
* `tcane_plotting_make_ALL`: puts it all together–we give it a storm and forecast ID and tell it where to look, and it loads all the data and calls the functions we need to make the plots. This function takes the following steps:
  * Creates output directories for TCANE visualization (`Figures/{storm_id}/`), if needed; 
  * Reads in relevant TCANE datafiles (`TCANE_input`, `TCANE_output`, and `TCANE_climo`). We use [`climo_to_df`, `read_in_TCANE`] functions located in `tcane_data_funcs.py`;  
  * Reads in matching real-time b-deck and e-deck files that correspond to `storm_ID`. `get_bdecks` and `get_edeck _probs` functions are located in `bdeck_edeck_funcs.py`; 
  * Checks to see if late forecasts are available (`skip_late`); 
  * Early forecast plots:
    * Calculates TCANE distributions from TCANE output for early forecasts;  (`make_TCANE_dists_pdf_cdf` function located in `tcane_plotting.py`); 
    * Gets relevant data for TCANE intensity plots from TCANE output; also creates relevant data using TCANE_climo for plots where we want to compare the current TCANE forecast to climatological TCANE forecast (`make_all_plotting_data` function located in `tcane_plotting.py`);   
    * Makes the TCANE intensity forecast plots (`TCANE_plots_intensity_forecasts`); 
    * Makes the TCANE rapid intensification and Category 1, etc probability plots (`TCANE_plots_cat_and_RI_plots`); 
    * Gets the relevant data for TCANE track forecast plots from TCANE output and from TCANE_climo output (`get_plot_vars_TRACK` from `tcane_track_plotting.py`); 
    * Makes TCANE track forecast plots using `TCANE_track_plots_with_climo` (TCANE forecasts for 66.67th-pctile and 95th pctile; from `tcane_track_plotting.py`); and the full TCANE track plot distributions (`TCANE_track_plot_all`, located in `tcane_track_plotting.py`);
  * Late forecast plots: If we have determined that late forecasts are available (`skip_late = False`), we make the same plots for the late forecasts. 

`TCANE_plots_intensity_forecasts`: This function calls `make_pctile_plot` from `tcane_plotting.pt` to create the TCANE intensity forecast plots. 

`TCANE_plots_cat_and_RI_plots`: This function calls `get_cat_probs`, `plot_RI`, and `plot_RI_bar` (all from `tcane_plotting.py`)  to create the TCANE cat probability and rapid intensification plots. Note that `plot_RI` is called twice–once with the e-deck data and once without. Feel free to disable one version of these plots. 

`TCANE_track_plots_with_climo`: This function calls `make_track_plt_climo` (from `tcane_track_plotting.py`) to create the climatological track plots for TCANE forecasts. We make two versions of this plot–one at the 66.667 percentile (default), and one at the 95 percentile. Users can change this by changing the `cmax` argument

<b>Reading in bdeck and edeck files</b>: `bdeck_edeck_funcs.py`: This file contains the functions needed to read in, parse, and save the realtime b-deck files (`get_bdecks`) and e-deck files (`get_edeck_probs`). The relevant b-deck and e-deck files for the given storm number are exported to the rest of the plotting code as Pandas dataframes. 

<i>Note</i>: The default configuration in `get_edeck_probs` is to read in the SHIPS-RII forecasts (`RIOD`) and the SHIPS RI Consensus forecasts (`RIOC`). If you want to add, remove, or change one of these products (say, you might want to include `DTOPS`), you can update the `tech_sel` argument in `get_edeck_probs`  

<b>Creating visualization data from TCANE forecasts</b>: `tcane_data_funcs.py`: This file contains several functions that read in the TCANE model output and calculate relevant quantities that we need for TCANE visualization. These functions do the following tasks: 
  * Read in TCANE data (`climo_to_df`, `read_in_TCANE`) and save it as a Pandas dataframe; 
  * Extract TCANE distribution parameters from TCANE output and save TCANE distribution as a new dataframe (`get_TCANE_distribution`);
  * Use TCANE distribution parameters to reconstruct the TCANE hyperbolic sinh-hyperbolic arcsine distribution (sinh-arcsinh, or SHASH, distribution), and export probability distribution as a Dataframe (`make_SHASH`);
  * Estimate specified percentiles from the TCANE data distribution (`calc_pctiles`);
  * Get probability density functions and cumulative density functions from TCANE distribution (`get_PDF_CDF`); 
  * Estimate rapid intensification probabilities, and the probabilities of achieving Category 1, 2, … 5 hurricane intensity for the TCANE forecast (`get_RI_info`)

<b>Creating TCANE graphical products, intensity</b>: `tcane_plotting`: This file contains several functions that create the various TCANE intensity forecast graphical products from the TCANE model output. These functions do the following tasks: 
  * `make_all_plotting_data`: This function is just a wrapper that calls various functions from `tcane_data_funcs` to create SHASH distributions, PDFs, CDFs, RI estimates, and so on; 
  * `make_TCANE_dists_pdf_cdf`: This function creates just the SHASH distribution and PDFs/CDFs for the TCANE forecast; 
  * [not used] `make_boxplot`: This function takes TCANE forecasts and makes a box plot comparing the early and late forecasts. This forecast product is not included in the current version of the TCANE suite but can be useful for debugging; 
  * `get_cat_probs`: This function creates a plot that compares the TCANE-forecasted probability of achieving {Cat 1…Cat 5} hurricane strength, and the climatological probability of a storm reaching that strength;  
  * `plot_RI`: This function creates a simple plot that compares the TCANE-forecasted probability of rapid intensification (RI) to the probability of RI forecast by other RI forecast products;  
  * `make_pctile_plot`: This function plots the median TCANE intensity forecast, and three levels of uncertainty: the interquartile range (25-75th percentiles), the realistic best and worst case (10th-90th percentiles), and the extreme cases (1-99th percentiles); 
  * `plot_RI_bar`: This function creates a bar plot that compares the TCANE RI forecast to other RI forecasts (e.g., SHIPS-RII) as well as the climatological probability of RI. The information about the climatological probability of RI is stored in `RI_percentages_TCANE_training.csv`. 

<b>Creating TCANE graphical products, track</b>: `tcane_track_plotting`: This file contains several functions that create the various TCANE track forecast graphical products from the TCANE model output. These functions do the following tasks: 
  * `plot_circle`, `plot_ellipse`: functions to plot a circle and ellipse with specified parameters;
  * `get_plot_vars_TRACK`: reformats TCANE track forecast variables to make plotting easier;
  * `get_plot_lims_fore`: gets the desired limits for TCANE track forecast plots in latitude and longitude so we can plot them on a map; 
  * `make_track_plt`: This function plots the ellipses of uncertainty on a map for TCANE track forecasts; shows full TCANE track forecast uncertainty distribution. Desired contours for display are specified but a user can change them if desired. The ellipses are calculated using functions in `mahalanobis.py` and `plots.py`. 
  * `make_track_plt_climo`: This function plots the uncertainty for the TCANE track forecast at a specified level (specified by `cmax`), as well as the corresponding forecast uncertainty for TCANE_climo. `cmax` is set to the 66.667 percentile by default but can be changed. The ellipses are calculated using functions in `mahalanobis.py` and `plots.py`. 

<b>Mathematical functions</b> for calculating TCANE track uncertainty ellipses: 
  * `plots.py`: This file contains several functions that calculate the ellipses of uncertainty for TCANE track forecasts. Modified from original TCANE codebase by Elizabeth Barnes, Randal Barnes, and Mark DeMaria (https://eartharxiv.org/repository/view/3470/). 
      * `set_plot_rc`, `adjust_spines`, `format_spines`, `draw_coastlines`: these are all just plot formatting functions
      * `plot_probability_ellipses`, `plot_probability_ellipses_vector`: Calculates elliptical TCANE track forecast uncertainty. This is calculated by estimating the Mahalanobis CDF for [x,y] using bivariate normal distributions. These functions call `plot_cdf`, `compute_cdf` from `mahalanobis.py`
  * `mahalnobis.py`: This file contains the actual mathematical formulae for calculating the TCANE track forecast uncertainty using the Mahalanobis CDF for a given [x,y], assuming a bivariate normal distribution. Modified from original TCANE codebase by Elizabeth Barnes, Randal Barnes, and Mark DeMaria (https://eartharxiv.org/repository/view/3470/). 
      * `compute_cdf`: computes Mahalanobis CDF using x and y components (and their uncertainties) of TCANE track forecasts; 
      * `compute_rsqr`: computes Mahalanobis R2 using x and y components (and their uncertainties) of TCANE track forecasts; 
      * `plot_cdf`: plots Mahalanobis CDF
  
<b>Additional data formatting functions</b>: 
  * `compute_predictions.py`: This file contains several functions that reformat the TCANE model output and predictions. Copied from original TCANE codebase by Elizabeth Barnes, Randal Barnes, and Mark DeMaria (https://eartharxiv.org/repository/view/3470/)
  * `data_info.py`: This file contains a function to reformat storm details to optimize plotting code. Copied from original TCANE codebase by Elizabeth Barnes, Randal Barnes, and Mark DeMaria (https://eartharxiv.org/repository/view/3470/). 

<b>Additional files</b>: `RI_percentages_TCANE_training.csv`: this is a .csv file that contains the climatological RI probabilities for each basin at a given probability threshold. 
