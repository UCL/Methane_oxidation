# Methane_oxidation

## 📌 Description
This repository contains code for identification of kinetic models for methane complete oxidation using statistical parameter estimation and model-based design of experiments methods.

## 🔧 Features
- Kinetic models for complete oxidation of methane in a micro-packed bed reactor, modelled as an isothermal plug flow reactor operated at steady state.
- Pressure drop model to estimate the average pressure along the packed bed reactor
- Model simulation
- Maximum likelihood parameter estimation implemented in Pyomo (https://pyomo.readthedocs.io/en/stable/explanation/modeling/dae.html)
- Probability of model adequacy expressed as probability not to reject a model assuming null hypothesis (the residuals of the model follows the distribution of the measurement error) is true
- Computation of first order parameter sensitivities (using perturbation analysis and finite difference approximation of derivatives)
- Computation of observed Fisher Information Matrix (FIM)
- Statistical hypothesis tests for model adequacy (chisquare goodness of fit test) and for parameter precision (t test)
- Model based design of experiments for model discrimination (Buzzi Ferraris criterion) and improving parameter precision (autonomous switching between these objectives)
- Data generation for model prediction density plots by propagating uncertainty of parameter estimates to uncertainty of response varaibles

## 💡 Usage Notes
The parameter estimation in this work was performed using Pyomo (https://www.pyomo.org/) with the IPOPT solver. To use IPOPT, download the executable from the official IPOPT GitHub releases page (https://github.com/coin-or/Ipopt/releases). 
If you are using Anaconda Python, follow these steps:
1. Download the zip file (Ipopt-3.14.16-win64-msvs2019-md.zip) from the link above.
2. Extract the contents of the ZIP file to your Anaconda "Library" folder (typically located inside your Anaconda installation path). 
3. In the Pyomo parameter estimation script, provide the full path to the IPOPT executable (e.g., ipopt.exe) when creating the solver instance, like this:
```python
solver = SolverFactory('ipopt', executable='C:/Path/To/Anaconda/Library/bin/ipopt.exe')
```
**Note**: Make sure the path matches where you have extarcted the solver.

