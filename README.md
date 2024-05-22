# Methane_oxidation
This repository contains code for identification of kinetic models for methane complete oxidation using statistical parameter estimation and model-based design of experiments methods. 

The parameter estimation involved in this work was carried out in Pyomo (https://www.pyomo.org/) using the IPOPT solver. To use the solver, please get the executable file from GitHub using the link (https://github.com/coin-or/Ipopt/releases). 
If you are using Anaconda Python, download the zip folder (Ipopt-3.14.16-win64-msvs2019-md.zip) from the shared link and extract it to the anaconda "Library" folder. Then, provide the path to the ipopt executable file within the extracted folder, when the solver instance is created in Pyomo.

This file is updated by Arun
