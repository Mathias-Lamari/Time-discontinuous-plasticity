# Time-discontinuous-plasticity

This repository contains the code used to simulate the time-discontinuous plasticity model introduced in the article "A time-discontinuous elasto-plasticity formalism to simulate instantaneous plastic flow bursts". The model, implemented in FEniCSX, introduces a plastic threshold parameter, $\Delta p_{\min}$, which captures the spatial and temporal localization of plastic flow in elastoplastic materials.

## Contents

+ `time_discontinuous_plasticity_fenicsx.py`: Contains the source code for the time-discontinuous plasticity model in FEniCSX
+ `Flat_specimen_refined_01.msh`: Contains the geometry used in the reference simulation
+ `averaged_data`: Folder where simulation statistics of stress and plastic strain are save. They are obtained by volume integration over the gauge length.

## Requirements

Before running the simulations, ensure that you have the following dependencies installed:

+ Python 3.10.10
+ NumPy: 1.24.3
+ SciPy: 1.11.4
+ ufl: 2023.1.1.post0
+ FEniCSX 0.6.0 for `time_discontinuous_plasticity_fenicsx_V0_6_0.py` OR FEniCSX 0.7.3 for `time_discontinuous_plasticity_fenicsx_V0_7_3.py`

Installation instructions for  can be found at the FEniCSX Documentation

## Videos and data sets

Videos and data sets of the simulation obtained from the time-discontinuous-plasticity simulation can be found in the Zenodo repository titled `Time-discontinuous plasticity, data and videos’ at [doi.org/10.5281/zenodo.13823869](https://doi.org/10.5281/zenodo.13823869).


