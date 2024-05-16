# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:53:08 2020

@author: p_aru
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import scipy.stats as stats
from openpyxl import load_workbook
import random
import pyomo.environ as py
from closedloopmain import MLE_pd
from closedloopmain import km1, km2, km3
from closedloopmain import residual
from closedloopmain import obs_FIM, obs_COR, pred_error
from closedloopmain import tvalue_fun, mprob1, mprob2
from closedloopmain import optikm1, optikm2, optikm3
from closedloopmain import mbdoepp, initialisation0_mbdoepp, initialisation1_mbdoepp
from closedloopmain import initialisation0_mbdoemd_BF, initialisation1_mbdoemd_BF, mbdoemd_BF

prng = 3
random.seed(prng)
np.random.seed(prng)
    

wb = load_workbook('factrdata1.xlsx') # Reading data file
ws1 = wb["DoE_D"]
row_count = 0
for row in ws1:
    if not any([cell.value == None for cell in row]):
        row_count += 1

n_prelim = row_count - 1
n_phi = 4 # number of design variables, excluding the outlet pressure
n_u = 5 # number of controls
n_y = 3 # number of measured responses
u_p = np.zeros([n_prelim,n_u])
y_meas = np.zeros([n_prelim,n_y])
Pin_meas = np.zeros(n_prelim)
    
for i in range(n_prelim):
    u_p[i,0] = ws1['B'+str(i + 2)].value
    u_p[i,1] = ws1['C'+str(i + 2)].value
    u_p[i,2] = ws1['E'+str(i + 2)].value
    u_p[i,3] = ws1['D'+str(i + 2)].value
    u_p[i,4] = ws1['F'+str(i + 2)].value
    y_meas[i,0] = ws1['I'+str(i + 2)].value
    y_meas[i,1] = ws1['H'+str(i + 2)].value
    y_meas[i,2] = ws1['G'+str(i + 2)].value
    Pin_meas[i] = ws1['J'+str(i + 2)].value
        
st = []
st += np.shape(u_p)[0] * [np.array([0, 0.01])]
    
ym = list(range(np.shape(u_p)[0]))
n_y = np.shape(y_meas)[1]
for i in range(np.shape(u_p)[0]):
    ym[i] = np.zeros([np.shape(st[i])[0],n_y])
ym_array = np.asarray(ym)
for i in range(np.shape(u_p)[0]):
    ym_array[i][0] = np.array([u_p[i][2],u_p[i][2]*u_p[i][3],0.0])
    ym_array[i][1] = y_meas[i]

lb_m1 = [0,0]
ub_m1 = [2e2,2e2]
ig_m1 = [6.9,7.3]

lb_m2 = [0,0,0,0,0,0]
ub_m2 = [2e2,2e2,2e2,2e2,2e2,2e2]
ig_m2 = [8.9,5.4,3.7,1.4,4.3,1.1]

lb_m3 = [0,0,0,0,0,0]
ub_m3 = [2e2,2e2,2e2,2e2,2e2,2e2]
ig_m3 = [2.0,9.2,5.6,3.5,10.6,9.0]
    
# NOTE: Measurement error covariance matrix
sigma_y = np.array([[0.00043**2, 0, 0], [0, 0.00202**2, 0], [0, 0, 0.00051**2]])
    
sigma_P = 0.005
    
pdtheta_hat = []
pdtheta_hat += [minimize_scalar(MLE_pd, bounds = (1e-6,1e6), method = 'bounded', 
                              args = (u_p,Pin_meas,sigma_P)).x]
    
alpha = 0.05
conf_level = 1.0 - alpha
y_cov = np.array([[0.00043**2, 0, 0], [0, 0.00202**2, 0], [0, 0, 0.00051**2]])
mexp = 20
    

# NOTE: Parameter estimation of pressure drop model
Pi = []
Pi += [minimize_scalar(MLE_pd, bounds = (1e-6,1e6), method = 'bounded', 
                              args = (u_p[0:mexp],Pin_meas[0:mexp],sigma_P)).x]

# NOTE: Calibration and posterior analysis of kinetic models
FIM1_obs, COV1_obs, COR1_obs, dof1, CI1, tval1 = [], [], [], [], [], []
FIM2_obs, COV2_obs, COR2_obs, dof2, CI2, tval2 = [], [], [], [], [], []
FIM3_obs, COV3_obs, COR3_obs, dof3, CI3, tval3 = [], [], [], [], [], []

# !!! One needs to put ipopt executable file in the path to use the solver
# NOTE: Parameter estimation using Pyomo model for power law kinetics
solver_m1 = py.SolverFactory('ipopt', executable ='C:/Users/Arun/anaconda3/Library/Ipopt-3.14.6-win64-msvs2019-md/bin/ipopt.exe')
optmodel_m1 = optikm1(u_p[0:mexp], ym_array[0:mexp], st[0:mexp], Pi[-1], ig_m1, lb_m1, ub_m1)
soln_m1 = solver_m1.solve(optmodel_m1)
est_m1 = np.array([py.value(optmodel_m1.theta[0]), py.value(optmodel_m1.theta[1])])
objf_m1 = py.value(optmodel_m1.objfun)
FIM1_obs += [obs_FIM(u_p[0:mexp],y_meas,km1,est_m1,Pi[-1])]
COV1_obs += [np.linalg.inv(FIM1_obs[-1])]
dof1 += [mexp * np.shape(y_meas)[1] - np.shape(est_m1)[0]]
CI1 += [tvalue_fun(est_m1,COV1_obs[-1],dof1[-1])[0]]
tval1 += [tvalue_fun(est_m1,COV1_obs[-1],dof1[-1])[1]]
tref1 = stats.t.ppf((1-alpha),dof1[-1])
COR1_obs += [obs_COR(COV1_obs[-1])]
sim1 = residual(u_p[0:mexp],est_m1,km1,y_meas[0:mexp],Pi[-1])[2]
resid1 = residual(u_p[0:mexp],est_m1,km1,y_meas[0:mexp],Pi[-1])[0]
nresid1 = residual(u_p[0:mexp],est_m1,km1,y_meas[0:mexp],Pi[-1])[1]
pred_cov1 = np.zeros([mexp,np.shape(y_meas)[1]])
for i in range(mexp):
    pred_cov1[i] = pred_error(u_p[i],y_meas,km1,est_m1,COV1_obs[-1],Pi[-1],dof1[-1])[1]


# NOTE: Parameter estimation using Pyomo model for LHHW model
solver_m2 = py.SolverFactory('ipopt', executable ='C:/Users/Arun/anaconda3/Library/Ipopt-3.14.6-win64-msvs2019-md/bin/ipopt.exe')
optmodel_m2 = optikm2(u_p[0:mexp], ym_array[0:mexp], st[0:mexp], Pi[-1], ig_m2, lb_m2, ub_m2)
soln_m2 = solver_m2.solve(optmodel_m2)
est_m2 = np.array([py.value(optmodel_m2.theta[0]), py.value(optmodel_m2.theta[1]), py.value(optmodel_m2.theta[2]), py.value(optmodel_m2.theta[3]), py.value(optmodel_m2.theta[4]), py.value(optmodel_m2.theta[5])])
objf_m2 = py.value(optmodel_m2.objfun)
FIM2_obs += [obs_FIM(u_p[0:mexp],y_meas,km2,est_m2,Pi[-1])]
COV2_obs += [np.linalg.inv(FIM2_obs[-1])]
dof2 += [mexp * np.shape(y_meas)[1] - np.shape(est_m2)[0]]
CI2 += [tvalue_fun(est_m2,COV2_obs[-1],dof2[-1])[0]]
tval2 += [tvalue_fun(est_m2,COV2_obs[-1],dof2[-1])[1]]
tref2 = stats.t.ppf((1-alpha),dof2[-1])
COR2_obs += [obs_COR(COV2_obs[-1])]
sim2 = residual(u_p[0:mexp],est_m2,km2,y_meas[0:mexp],Pi[-1])[2]
resid2 = residual(u_p[0:mexp],est_m2,km2,y_meas[0:mexp],Pi[-1])[0]
nresid2 = residual(u_p[0:mexp],est_m2,km2,y_meas[0:mexp],Pi[-1])[1]
pred_cov2 = np.zeros([mexp,np.shape(y_meas)[1]])
for i in range(mexp):
    pred_cov2[i] = pred_error(u_p[i],y_meas,km2,est_m2,COV2_obs[-1],Pi[-1],dof2[-1])[1]
    
    
# NOTE: Parameter estimation using Pyomo model for MVK model
solver_m3 = py.SolverFactory('ipopt', executable ='C:/Users/Arun/anaconda3/Library/Ipopt-3.14.6-win64-msvs2019-md/bin/ipopt.exe')
optmodel_m3 = optikm3(u_p[0:mexp], ym_array[0:mexp], st[0:mexp], Pi[-1], ig_m3, lb_m3, ub_m3)
soln_m3 = solver_m3.solve(optmodel_m3)
est_m3 = np.array([py.value(optmodel_m3.theta[0]), py.value(optmodel_m3.theta[1]), py.value(optmodel_m3.theta[2]), py.value(optmodel_m3.theta[3]), py.value(optmodel_m3.theta[4]), py.value(optmodel_m3.theta[5])])
objf_m3 = py.value(optmodel_m3.objfun)
FIM3_obs += [obs_FIM(u_p[0:mexp],y_meas,km3,est_m3,Pi[-1])]
COV3_obs += [np.linalg.inv(FIM3_obs[-1])]
dof3 += [mexp * np.shape(y_meas)[1] - np.shape(est_m3)[0]]
CI3 += [tvalue_fun(est_m3,COV3_obs[-1],dof3[-1])[0]]
tval3 += [tvalue_fun(est_m3,COV3_obs[-1],dof3[-1])[1]]
COR3_obs += [obs_COR(COV3_obs[-1])]
sim3 = residual(u_p[0:mexp],est_m3,km3,y_meas[0:mexp],Pi[-1])[2]
sim31 = np.zeros([mexp,np.shape(y_meas)[1]])
for i in range(mexp):
    sim31[i] = pred_error(u_p[i],y_meas,km3,est_m3,COV3_obs[-1],Pi[-1],dof3[-1])[0]
resid3 = residual(u_p[0:mexp],est_m3,km3,y_meas[0:mexp],Pi[-1])[0]
nresid3 = residual(u_p[0:mexp],est_m3,km3,y_meas[0:mexp],Pi[-1])[1]
tref3 = stats.t.ppf((1-alpha),dof3[-1])
pred_cov3 = np.zeros([mexp,np.shape(y_meas)[1]])
for i in range(mexp):
    pred_cov3[i] = pred_error(u_p[i],y_meas,km3,est_m3,COV3_obs[-1],Pi[-1],dof3[-1])[1]

# NOTE: Chi-square goodness-of-fit test
refchisq1 = stats.chi2.ppf((conf_level),dof1[-1])
refchisq2 = stats.chi2.ppf((conf_level),dof2[-1])
refchisq3 = stats.chi2.ppf((conf_level),dof3[-1])

# NOTE: Computation of probability of model correctness
obspr1 = []
obspr1 += [mprob1([objf_m1,objf_m2,objf_m3],[dof1[-1],dof2[-1],dof3[-1]])[1]]
    
obspr2 = []
obspr2 += [mprob2([objf_m1,objf_m2,objf_m3],[dof1[-1],dof2[-1],dof3[-1]])]

n_dexp = 1 # Number of experiments to be designed using MBDoE method

# NOTE: MBDoE for model discrimination (between model 2 and 3)
ig_md1, b_md1 = initialisation1_mbdoemd_BF(y_meas,km2,km3,est_m2,est_m3,FIM2_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_u)#,prng)
ig_md0, b_md0 = initialisation0_mbdoemd_BF(y_meas,km2,km3,est_m2,est_m3,FIM2_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_u,prng)
sol_md1 = minimize(mbdoemd_BF,ig_md1,method = 'SLSQP', bounds = (b_md1), args = (y_meas,km2,km3,est_m2,est_m3,FIM2_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_phi))
sol_md0 = minimize(mbdoemd_BF,ig_md0,method = 'SLSQP', bounds = (b_md0), args = (y_meas,km2,km3,est_m2,est_m3,FIM2_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_phi))

# NOTE: MBDoE for parameter precision of model 3
ig_pp1, b_pp1 = initialisation1_mbdoepp(y_meas,km3,est_m3,FIM3_obs[-1],Pi[-1],n_dexp,n_u)#,prng)
ig_pp0, b_pp0 = initialisation0_mbdoepp(y_meas,km3,est_m3,FIM3_obs[-1],Pi[-1],n_dexp,n_u,prng)
sol_pp1 = minimize(mbdoepp, ig_pp1, method = 'SLSQP', bounds = (b_pp1), args = (y_meas,km3,est_m3,FIM3_obs[-1],Pi[-1],n_dexp,n_phi))
sol_pp0 = minimize(mbdoepp, ig_pp0, method = 'SLSQP', bounds = (b_pp0), args = (y_meas,km3,est_m3,FIM3_obs[-1],Pi[-1],n_dexp,n_phi))
