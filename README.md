# MGCM_Public

This repository contains the utilities needed to emulate the NASA Ames Mars Global Climate Model (MGCM) using Gaussian Process emulation.

[![DOI](https://zenodo.org/badge/562292845.svg)](https://zenodo.org/badge/latestdoi/562292845)

## Utilities

The `Emulator_Utils` directory contains four files used to collect data from the MGCM and train the emulator.

`Emulator_Utils/ExecutableBot.py` overwrites the Microphy.f file in the MGCM to modify the lifted dust effective radius parameter as per the paper. The MGCM executable is then created and saved to a path of the user's choosing by modifying the file parameters. Python Dependencies: NumPy and shutil.

`Emulator_Utils/GCMBot.py` runs the executables. The user may choose how many are run at any given time. Python Dependencies: psutil and shutil.

`Emulator_Utils/GCMConvert.py` converts the output of the MGCM to netcdf4 format. Python Dependencies: psutil, shutil, and CAP (CAP Installation: https://www.nasa.gov/sites/default/files/atoms/files/cap_install_0.pdf)

`Emulator_Utils/Emulate.py` trains the emulator and performs the emulator over a selection. Python Dependencies: sklearn, numba, h5py, and numpy.

The `Figures/Utils` directory contains four files that test and plot the results found in the paper.

`Figures/Utils/forward_error_table.py` outputs the data that was used to create Table1.

`Figures/Utils/PlotRelativeError.py` plots the forward error graph that was used in the paper.

`Figures/Utils/BWE_Combined.py` creates the combined backward error analysis plots. Python Dependencies: matplotlib, numpy, h5py, and sklearn.

`Figures/Utils/BWE_Plots.py` creates the remaining plots that were used in the paper. Slight modifications to the code are required for each type. Python Dependencies: matplotlib, numpy, h5py, and sklearn.

## Getting Started

1. Install the MGCM
2. Create the executables using `Emulator_Utils/ExecutableBot.py` as described
3. Run the simulations using `Emulator_Utils/GCMBot.py`
4. Convert the output to usable data via `Emulator_Utils/GCMConvert.py`
5. Train and run the emulator on outputs of interest via `Emulator_Utils/Emulate.py`
6. Plot the results using `Figures/Utils/BWE_Combined.py`, `Figures/Utils/BWE_Plots.py`, `Figures/Utils/forward_error_table.py`, and `Figures/Utils/PlotRelativeError.py` 
