# tcane_rt_viz
This code makes real-time graphical output for the TCANE model. After you have cloned this repo using `git clone`, navigate to the correct directory (`tcane_realtime_plotting`). 

1. Set up the environment file using `conda`: `conda env create -f environment.yml` (note that creating the environment might take a few minutes; don't panic! You only have to do this part once). This environment can only be used to run the realtime plotting code–we can’t train the model in this environment.

2. To activate the environment, run `conda activate tcane_rt`. To deactivate, the command is `conda deactivate`. The Python environment will need to be activated before running the plotting code. It should not affect any Fortran operations.

3. Make sure your directory structure is correct: You should have a main directory that contains a Python script (*.py), a Jupyter notebook (*.ipynb), the environment file (*.yml), and two subdirectories, `utils` and `Figures`. `utils` contains all of the functions and subroutines used to make the TCANE visuals. A more detailed description of what each script in `utils` contains is at the end of this readme. If the `Figures` directory doesn’t exist yet, don’t worry, the plotting code will create it for you.

4.  The plotting code needs realtime TCANE input files and output files, and TCANE climatology, to run. The plotting code can be run from the command line with the following inputs: `python run_TCANE_rt_plotting.py STORMID FOREDATE`
  * `STORMID` contains the ATCFID of the desired storm, in the form `BBNNYYYY`, where `BB` is the 2-letter basin abbreviation, `NN` is the 2 character storm number, and `YYYY` is the 4 character year. [string]
  * `FOREDATE` contains the forecast date of the desired storm in the form `MMDDHH`. [string]

  In addition, the plotting code needs to know where the TCANE files are located. The user needs to specify the relevant directories for TCANE input, output, and climatology files; we also need to specify the relevant directories for the real-time b-deck and e-deck files. These directories are specified within the Python script 
  
  * `INPUT_DIR` is the directory where the TCANE input files are located.
  * `OUTPUT_DIR` is the directory where the TCANE output files are located.
  * `BDECK_DIR` is the directory where b-deck and e-deck files are located.




