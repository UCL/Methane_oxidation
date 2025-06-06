# Methane_oxidation

## 📌 Description
This repository contains code for identification of kinetic models for methane complete oxidation using statistical parameter estimation and model-based design of experiments methods.

## 🔧 Features
- Kinetic models for complete oxidation of methane in a micro-packed bed reactor, modelled as an isothermal plug flow reactor operated at steady state.
- Pressure drop model to estimate the average pressure along the packed bed reactor
- Model simulation tools
- Maximum likelihood parameter estimation implemented in `Pyomo` (https://pyomo.readthedocs.io/en/stable/explanation/modeling/dae.html)
- `Probability of model adequacy` expressed as probability not to reject a model assuming null hypothesis (the residuals of the model follow the distribution of the measurement error) is true
- Computation of first order parameter sensitivities (using perturbation analysis and finite difference approximation of derivatives)
- Computation of observed `Fisher Information Matrix (FIM)` using sensitivity matrix
- Statistical hypothesis tests for `model adequacy` (chi-square goodness of fit test) and for `parameter precision` (_t_ test)
- Computation of `parameter covariance matrix` (approximated as the inverse of FIM) and of `parameter correlation matrix`
- Data generation to plot `confidence ellipse` for any pair of parameter estimates
- Model based design of experiments for `model discrimination` (Buzzi Ferraris criterion) and `improving parameter precision` (autonomous switching between these objectives)
- Data generation for `model prediction density plots` by propagating uncertainty in parameter estimates to uncertainty in response variables

## 💡 Usage Notes
The parameter estimation in this work was performed using `Pyomo` (https://www.pyomo.org/) with the `IPOPT` solver. To use IPOPT, download the executable from the official IPOPT GitHub releases page (https://github.com/coin-or/Ipopt/releases). 
If you are using Anaconda Python, follow these steps:
1. Download the zip file (Ipopt-3.14.16-win64-msvs2019-md.zip) from the link above.
2. Extract the contents of the ZIP file to your Anaconda "Library" folder (typically located inside your Anaconda installation path). 
3. In the Pyomo parameter estimation script, provide the full path to the IPOPT executable (e.g., ipopt.exe) when creating the solver instance, like this:
```python
solver = SolverFactory('ipopt', executable='C:/Path/To/Anaconda/Library/bin/ipopt.exe')
```
**Note**: Make sure the path matches where you have extracted the solver.

The Python functions implementing all the features mentioned above are provided in the file `closedloopmain.py`. The function calls to test these functions and execute the complete algorithm are included in `call_pe.py`. To test the code, use the data file `factrdata1.xlsx`, which is also the dataset used in the paper.

