# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:53:08 2020

@author: p_aru
"""

import numpy as np
#import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import scipy.stats as stats
from openpyxl import Workbook, load_workbook
import random
import scipy as sc

import pyomo.environ as py
import pyomo.dae as pyd

from closedloopmain import pdrop, MLE_pd, Pinmodel
from closedloopmain import km1, km2, km3, km2r
from closedloopmain import mle_fun, insilico_exp, residual, distributionplot1, distributionplot2, distributionplot3
from closedloopmain import sen_fun, obs_FIM, obs_COR, pred_error
from closedloopmain import tvalue_fun, mprob1, mprob2, confidence_ellipsoid1, confidence_ellipsoid31, confidence_ellipsoid32
from closedloopmain import tdomain, xmeas, controls, pcontrols
from closedloopmain import optikm1, optikm2, optikm3, optikm2r
from closedloopmain import mbdoepp, initialisation0_mbdoepp, initialisation1_mbdoepp
from closedloopmain import initialisation0_mbdoemd_BF, initialisation1_mbdoemd_BF, mbdoemd_BF

prng = 3
random.seed(prng)
np.random.seed(prng)
    
# wb = load_workbook('factrdata_latest.xlsx')
# ws1 = wb["DoE"]
wb = load_workbook('factrdata1.xlsx')
ws1 = wb["DoE_D"]
# ws1 = wb["DoE_A"]
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
# ig_m2 = [2.0,9.2,5.6,3.5,10.6,9.0]

lb_m2r = [0,0,0,0,0]
ub_m2r = [2e2,2e2,2e2,2e2,2e2]
ig_m2r = [8.9,5.4,3.7,1.4,4.3]

lb_m3 = [0,0,0,0,0,0]
ub_m3 = [2e2,2e2,2e2,2e2,2e2,2e2]
ig_m3 = [2.0,9.2,5.6,3.5,10.6,9.0]
# ig_m3 = [5.3,6.9,4.8,10.5,10.4,7.9]
# ig_m3 = [3.3,10.9,4.8,10.5,10.4,1.9]
    
    
sigma_y = np.array([[0.00043**2, 0, 0], [0, 0.00202**2, 0], [0, 0, 0.00051**2]])
    
sigma_P = 0.005
    
pdtheta_hat = []
pdtheta_hat += [minimize_scalar(MLE_pd, bounds = (1e-6,1e6), method = 'bounded', 
                              args = (u_p,Pin_meas,sigma_P)).x]
    
alpha = 0.05
conf_level = 1.0 - alpha
y_cov = np.array([[0.00043**2, 0, 0], [0, 0.00202**2, 0], [0, 0, 0.00051**2]])
'''
 Parameter estimation in Pyomo
'''
mexp = 20
    
    ### global method by splitting dataset and bounding solution ###
    
Pi = []
FIM1_obs, COV1_obs, COR1_obs, dof1, CI1, tval1 = [], [], [], [], [], []
FIM2_obs, COV2_obs, COR2_obs, dof2, CI2, tval2 = [], [], [], [], [], []
# FIM2r_obs, COV2r_obs, COR2r_obs, dof2r, CI2r, tval2r = [], [], [], [], [], []
FIM3_obs, COV3_obs, COR3_obs, dof3, CI3, tval3 = [], [], [], [], [], []
    
Pi += [minimize_scalar(MLE_pd, bounds = (1e-6,1e6), method = 'bounded', 
                              args = (u_p[0:mexp],Pin_meas[0:mexp],sigma_P)).x]
    
    
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

# return np.linalg.eigvals(FIM1_obs[-1]), np.diag(FIM1_obs[-1]), np.linalg.eigvals(FIM2_obs[-1]), np.diag(FIM2_obs[-1]), np.linalg.eigvals(FIM3_obs[-1]), FIM3_obs[-1], np.diag(COV3_obs[-1])



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
    
    
# solver_m2r = py.SolverFactory('ipopt', executable ='C:/Users/Arun/anaconda3/Library/Ipopt-3.14.6-win64-msvs2019-md/bin/ipopt.exe')
# optmodel_m2r = optikm2r(u_p[0:mexp], ym_array[0:mexp], st[0:mexp], Pi[-1], ig_m2r, lb_m2r, ub_m2r)
# soln_m2r = solver_m2r.solve(optmodel_m2r)
# est_m2r = np.array([py.value(optmodel_m2r.theta[0]), py.value(optmodel_m2r.theta[1]), py.value(optmodel_m2r.theta[2]), py.value(optmodel_m2r.theta[3]), py.value(optmodel_m2r.theta[4])])
# objf_m2r = py.value(optmodel_m2r.objfun)
    
# FIM2r_obs += [obs_FIM(u_p[0:mexp],y_meas,km2r,est_m2r,Pi[-1])]
# COV2r_obs += [np.linalg.inv(FIM2r_obs[-1])]
# dof2r += [mexp * np.shape(y_meas)[1] - np.shape(est_m2r)[0]]
# CI2r += [tvalue_fun(est_m2r,COV2r_obs[-1],dof2r[-1])[0]]
# tval2r += [tvalue_fun(est_m2r,COV2r_obs[-1],dof2r[-1])[1]]
# tref2r = stats.t.ppf((1-alpha),dof2r[-1])
# COR2r_obs += [obs_COR(COV2r_obs[-1])]
# sim2r = residual(u_p[0:mexp],est_m2r,km2r,y_meas[0:mexp],Pi[-1])[2]
# resid2r = residual(u_p[0:mexp],est_m2r,km2r,y_meas[0:mexp],Pi[-1])[1]
# pred_cov2r = np.zeros([mexp,np.shape(y_meas)[1]])
# for i in range(mexp):
#     pred_cov2r[i] = pred_error(u_p[i],y_meas,km2r,est_m2r,COV2r_obs[-1],Pi[-1],dof2r[-1])[1]
    
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

refchisq1 = stats.chi2.ppf((conf_level),dof1[-1])
refchisq2 = stats.chi2.ppf((conf_level),dof2[-1])
# refchisq2r = stats.chi2.ppf((conf_level),dof2r[-1])
refchisq3 = stats.chi2.ppf((conf_level),dof3[-1])
    
obspr1 = []
obspr1 += [mprob1([objf_m1,objf_m2,objf_m3],[dof1[-1],dof2[-1],dof3[-1]])[1]]
    
obspr2 = []
obspr2 += [mprob2([objf_m1,objf_m2,objf_m3],[dof1[-1],dof2[-1],dof3[-1]])]
    
xval1, yval1 = confidence_ellipsoid1(est_m1,COV1_obs[-1],0.05)
xval31, yval31 = confidence_ellipsoid31(est_m3,COV3_obs[-1],0.05)
xval32, yval32 = confidence_ellipsoid32(est_m3,COV3_obs[-1],0.05)

# truetheta = [5.31, 6.96, 4.88, 10.50, 10.44, 7.95] # we have assumed this as the true values for model parameters
# truepdtheta = [np.array(0.0073)] # assumed that this is the true value of the pressure drop paraneter

# newdata = insilico_exp(u_p[0],truetheta,km3,truepdtheta,y_cov)

# plot1 = distributionplot1(u_p[0],est_m3,km3,Pi[-1],CI3[-1])
plot2 = list()
for i in range(mexp):
    plot2 += [distributionplot2(u_p[i],y_meas,km2,est_m2,COV2_obs[-1],Pi[-1])]
plot3 = list()
for i in range(mexp):
    plot3 += [distributionplot2(u_p[i],y_meas,km3,est_m3,COV3_obs[-1],Pi[-1])]

plot4 = list()
for i in range(mexp):
    plot4 += [distributionplot3(u_p[i],est_m3,km3,Pi[-1],COV3_obs[-1])]
    
plot5 = list()
for i in range(mexp):
    plot5 += [distributionplot3(u_p[i],est_m2,km2,Pi[-1],COV2_obs[-1])]

wb = Workbook()
ws = wb.active
ws.title = 'PE results'

ws.cell(row = 1, column = 1, value = ('Model 1'))
ws.cell(row = 2, column = 1, value = ('Estimate'))
ws.cell(row = 2, column = 2, value = ('CI'))
ws.cell(row = 2, column = 3, value = ('tvalue'))
ws.cell(row = 2, column = 4, value = ('Chisquare'))
ws.cell(row = 2, column = 5, value = ('Probability'))

ws.cell(row = 1, column = 7, value = ('Model 2'))
ws.cell(row = 2, column = 7, value = ('Estimate'))
ws.cell(row = 2, column = 8, value = ('CI'))
ws.cell(row = 2, column = 9, value = ('tvalue'))
ws.cell(row = 2, column = 10, value = ('Chisquare'))
ws.cell(row = 2, column = 11, value = ('Probability'))
   
ws.cell(row = 1, column = 13, value = ('Model 3'))
ws.cell(row = 2, column = 13, value = ('Estimate'))
ws.cell(row = 2, column = 14, value = ('CI'))
ws.cell(row = 2, column = 15, value = ('tvalue'))
ws.cell(row = 2, column = 16, value = ('Chisquare'))
ws.cell(row = 2, column = 17, value = ('Probability'))
    

for i in range(np.shape(est_m1)[0]):
    ws.cell(row = 3+i, column = 1, value = (est_m1[i]))
    ws.cell(row = 3+i, column = 2, value = (CI1[-1][i]))
    ws.cell(row = 3+i, column = 3, value = (tval1[-1][i]))
ws.cell(row = 3+np.shape(est_m1)[0], column = 3, value = (tref1))
ws.cell(row = 3, column = 4, value = (objf_m1))
ws.cell(row = 4, column = 4, value = (refchisq1))
ws.cell(row = 3, column = 5, value = (obspr1[-1][0]))
for i in range(np.shape(est_m2)[0]):
    ws.cell(row = 3+i, column = 7, value = (est_m2[i]))
    ws.cell(row = 3+i, column = 8, value = (CI2[-1][i]))
    ws.cell(row = 3+i, column = 9, value = (tval2[-1][i]))
ws.cell(row = 3+np.shape(est_m2)[0], column = 9, value = (tref2))
ws.cell(row = 3, column = 10, value = (objf_m2))
ws.cell(row = 4, column = 10, value = (refchisq2))
ws.cell(row = 3, column = 11, value = (obspr1[-1][1]))
for i in range(np.shape(est_m3)[0]):
    ws.cell(row = 3+i, column = 13, value = (est_m3[i]))
    ws.cell(row = 3+i, column = 14, value = (CI3[-1][i]))
    ws.cell(row = 3+i, column = 15, value = (tval3[-1][i]))
ws.cell(row = 3+np.shape(est_m3)[0], column = 15, value = (tref3))
ws.cell(row = 3, column = 16, value = (objf_m3))
ws.cell(row = 4, column = 16, value = (refchisq3))
ws.cell(row = 3, column = 17, value = (obspr1[-1][2]))
    
ws1 = wb.create_sheet('Correlation matrix')
ws1.cell(row = 1, column = 1, value = ('Model 1'))
ws1.cell(row = 2, column = 1, value = ('column1'))
ws1.cell(row = 2, column = 2, value = ('column2'))
   
ws1.cell(row = 1, column = 5, value = ('Model 2'))
ws1.cell(row = 2, column = 5, value = ('column1'))
ws1.cell(row = 2, column = 6, value = ('column2'))
ws1.cell(row = 2, column = 7, value = ('column3'))
ws1.cell(row = 2, column = 8, value = ('column4'))
ws1.cell(row = 2, column = 9, value = ('column5'))
ws1.cell(row = 2, column = 10, value = ('column6'))
    
ws1.cell(row = 1, column = 13, value = ('Model 3'))
ws1.cell(row = 2, column = 13, value = ('column1'))
ws1.cell(row = 2, column = 14, value = ('column2'))
ws1.cell(row = 2, column = 15, value = ('column3'))
ws1.cell(row = 2, column = 16, value = ('column4'))
ws1.cell(row = 2, column = 17, value = ('column5'))
ws1.cell(row = 2, column = 18, value = ('column6'))
    
for i in range(np.shape(est_m1)[0]):
    ws1.cell(row = 3+i, column = 1, value = (COR1_obs[-1][:,0][i]))
    ws1.cell(row = 3+i, column = 2, value = (COR1_obs[-1][:,1][i]))
for i in range(np.shape(est_m2)[0]):
    ws1.cell(row = 3+i, column = 5, value = (COR2_obs[-1][:,0][i]))
    ws1.cell(row = 3+i, column = 6, value = (COR2_obs[-1][:,1][i]))
    ws1.cell(row = 3+i, column = 7, value = (COR2_obs[-1][:,2][i]))
    ws1.cell(row = 3+i, column = 8, value = (COR2_obs[-1][:,3][i]))
    ws1.cell(row = 3+i, column = 9, value = (COR2_obs[-1][:,4][i]))
    ws1.cell(row = 3+i, column = 10, value = (COR2_obs[-1][:,5][i]))
for i in range(np.shape(est_m3)[0]):
    ws1.cell(row = 3+i, column = 13, value = (COR3_obs[-1][:,0][i]))
    ws1.cell(row = 3+i, column = 14, value = (COR3_obs[-1][:,1][i]))
    ws1.cell(row = 3+i, column = 15, value = (COR3_obs[-1][:,2][i]))
    ws1.cell(row = 3+i, column = 16, value = (COR3_obs[-1][:,3][i]))
    ws1.cell(row = 3+i, column = 17, value = (COR3_obs[-1][:,4][i]))
    ws1.cell(row = 3+i, column = 18, value = (COR3_obs[-1][:,5][i]))

ws2 = wb.create_sheet('Confidence ellipse')
ws2.cell(row = 1, column = 1, value = ('Exp. No.'))
ws2.cell(row = 1, column = 2, value = ('Model'))
ws2.cell(row = 1, column = 3, value = ('x value'))
ws2.cell(row = 1, column = 4, value = ('y value'))
    
ws2.cell(row = 1, column = 7, value = ('Exp. No.'))
ws2.cell(row = 1, column = 8, value = ('Model'))
ws2.cell(row = 1, column = 9, value = ('x value'))
ws2.cell(row = 1, column = 10, value = ('y value'))

ws2.cell(row = 1, column = 13, value = ('Exp. No.'))
ws2.cell(row = 1, column = 14, value = ('Model'))
ws2.cell(row = 1, column = 15, value = ('x value'))
ws2.cell(row = 1, column = 16, value = ('y value'))
    
ws2.cell(row = 2, column = 1, value = (mexp))
ws2.cell(row = 2, column = 2, value = ('Model 1'))
ws2.cell(row = 2, column = 7, value = (mexp))
ws2.cell(row = 2, column = 8, value = ('Model 3_31'))
ws2.cell(row = 2, column = 13, value = (mexp))
ws2.cell(row = 2, column = 14, value = ('Model 3_32'))
for i in range(np.shape(xval31)[0]):
    ws2.cell(row = 2+i, column = 3, value = (xval1[i]))
    ws2.cell(row = 2+i, column = 4, value = (yval1[i]))
    ws2.cell(row = 2+i, column = 9, value = (xval31[i]))
    ws2.cell(row = 2+i, column = 10, value = (yval31[i]))
    ws2.cell(row = 2+i, column = 15, value = (xval32[i]))
    ws2.cell(row = 2+i, column = 16, value = (yval32[i]))

ws3 = wb.create_sheet('Simulation')
ws3.cell(row = 1, column = 1, value = ('Model 1'))
ws3.cell(row = 2, column = 1, value = ('Experiment'))
ws3.cell(row = 2, column = 2, value = ('Methane'))
ws3.cell(row = 2, column = 3, value = ('Oxygen'))
ws3.cell(row = 2, column = 4, value = ('Carbon dioxide'))

ws3.cell(row = 1, column = 7, value = ('Model 2'))
ws3.cell(row = 2, column = 7, value = ('Experiment'))
ws3.cell(row = 2, column = 8, value = ('Methane'))
ws3.cell(row = 2, column = 9, value = ('Oxygen'))
ws3.cell(row = 2, column = 10, value = ('Carbon dioxide'))

ws3.cell(row = 1, column = 13, value = ('Model 3'))
ws3.cell(row = 2, column = 13, value = ('Experiment'))
ws3.cell(row = 2, column = 14, value = ('Methane'))
ws3.cell(row = 2, column = 15, value = ('Oxygen'))
ws3.cell(row = 2, column = 16, value = ('Carbon dioxide'))


for i in range(mexp):
    ws3.cell(row = 3+i, column = 1, value = (i+1))
    ws3.cell(row = 3+i, column = 2, value = (sim1[:,0][i]))
    ws3.cell(row = 3+i, column = 3, value = (sim1[:,1][i]))
    ws3.cell(row = 3+i, column = 4, value = (sim1[:,2][i]))
    
    ws3.cell(row = 3+i, column = 7, value = (i+1))
    ws3.cell(row = 3+i, column = 8, value = (sim2[:,0][i]))
    ws3.cell(row = 3+i, column = 9, value = (sim2[:,1][i]))
    ws3.cell(row = 3+i, column = 10, value = (sim2[:,2][i]))
    
    ws3.cell(row = 3+i, column = 13, value = (i+1))
    ws3.cell(row = 3+i, column = 14, value = (sim3[:,0][i]))
    ws3.cell(row = 3+i, column = 15, value = (sim3[:,1][i]))
    ws3.cell(row = 3+i, column = 16, value = (sim3[:,2][i]))
    
ws4 = wb.create_sheet('Residual distribution')
ws4.cell(row = 1, column = 1, value = ('Model 1 normalised'))
ws4.cell(row = 2, column = 1, value = ('Experiment'))
ws4.cell(row = 2, column = 2, value = ('Methane'))
ws4.cell(row = 2, column = 3, value = ('Oxygen'))
ws4.cell(row = 2, column = 4, value = ('Carbon dioxide'))

ws4.cell(row = 1, column = 7, value = ('Model 2 normalised'))
ws4.cell(row = 2, column = 7, value = ('Experiment'))
ws4.cell(row = 2, column = 8, value = ('Methane'))
ws4.cell(row = 2, column = 9, value = ('Oxygen'))
ws4.cell(row = 2, column = 10, value = ('Carbon dioxide'))

ws4.cell(row = 1, column = 13, value = ('Model 3 normalised'))
ws4.cell(row = 2, column = 13, value = ('Experiment'))
ws4.cell(row = 2, column = 14, value = ('Methane'))
ws4.cell(row = 2, column = 15, value = ('Oxygen'))
ws4.cell(row = 2, column = 16, value = ('Carbon dioxide'))

ws4.cell(row = 1, column = 18, value = ('Model 1'))
ws4.cell(row = 2, column = 18, value = ('Experiment'))
ws4.cell(row = 2, column = 19, value = ('Methane'))
ws4.cell(row = 2, column = 20, value = ('Oxygen'))
ws4.cell(row = 2, column = 21, value = ('Carbon dioxide'))

ws4.cell(row = 1, column = 24, value = ('Model 2'))
ws4.cell(row = 2, column = 24, value = ('Experiment'))
ws4.cell(row = 2, column = 25, value = ('Methane'))
ws4.cell(row = 2, column = 26, value = ('Oxygen'))
ws4.cell(row = 2, column = 27, value = ('Carbon dioxide'))

ws4.cell(row = 1, column = 30, value = ('Model 3'))
ws4.cell(row = 2, column = 30, value = ('Experiment'))
ws4.cell(row = 2, column = 31, value = ('Methane'))
ws4.cell(row = 2, column = 32, value = ('Oxygen'))
ws4.cell(row = 2, column = 33, value = ('Carbon dioxide'))


for i in range(mexp):
    ws4.cell(row = 3+i, column = 18, value = (i+1))
    ws4.cell(row = 3+i, column = 19, value = (resid1[:,0][i]))
    ws4.cell(row = 3+i, column = 20, value = (resid1[:,1][i]))
    ws4.cell(row = 3+i, column = 21, value = (resid1[:,2][i]))

    ws4.cell(row = 3+i, column = 24, value = (i+1))
    ws4.cell(row = 3+i, column = 25, value = (resid2[:,0][i]))
    ws4.cell(row = 3+i, column = 26, value = (resid2[:,1][i]))
    ws4.cell(row = 3+i, column = 27, value = (resid2[:,2][i]))
    
    ws4.cell(row = 3+i, column = 30, value = (i+1))
    ws4.cell(row = 3+i, column = 31, value = (resid3[:,0][i]))
    ws4.cell(row = 3+i, column = 32, value = (resid3[:,1][i]))
    ws4.cell(row = 3+i, column = 33, value = (resid3[:,2][i]))
    
    ws4.cell(row = 3+i, column = 1, value = (i+1))
    ws4.cell(row = 3+i, column = 2, value = (nresid1[:,0][i]))
    ws4.cell(row = 3+i, column = 3, value = (nresid1[:,1][i]))
    ws4.cell(row = 3+i, column = 4, value = (nresid1[:,2][i]))

    ws4.cell(row = 3+i, column = 7, value = (i+1))
    ws4.cell(row = 3+i, column = 8, value = (nresid2[:,0][i]))
    ws4.cell(row = 3+i, column = 9, value = (nresid2[:,1][i]))
    ws4.cell(row = 3+i, column = 10, value = (nresid2[:,2][i]))
    
    ws4.cell(row = 3+i, column = 13, value = (i+1))
    ws4.cell(row = 3+i, column = 14, value = (nresid3[:,0][i]))
    ws4.cell(row = 3+i, column = 15, value = (nresid3[:,1][i]))
    ws4.cell(row = 3+i, column = 16, value = (nresid3[:,2][i]))
    
ws5 = wb.create_sheet('Prediction variance')
ws5.cell(row = 1, column = 1, value = ('Model 1'))
ws5.cell(row = 2, column = 1, value = ('Experiment'))
ws5.cell(row = 2, column = 2, value = ('Methane'))
ws5.cell(row = 2, column = 3, value = ('Oxygen'))
ws5.cell(row = 2, column = 4, value = ('Carbon dioxide'))

ws5.cell(row = 1, column = 7, value = ('Model 2'))
ws5.cell(row = 2, column = 7, value = ('Experiment'))
ws5.cell(row = 2, column = 8, value = ('Methane'))
ws5.cell(row = 2, column = 9, value = ('Oxygen'))
ws5.cell(row = 2, column = 10, value = ('Carbon dioxide'))

ws5.cell(row = 1, column = 13, value = ('Model 3'))
ws5.cell(row = 2, column = 13, value = ('Experiment'))
ws5.cell(row = 2, column = 14, value = ('Methane'))
ws5.cell(row = 2, column = 15, value = ('Oxygen'))
ws5.cell(row = 2, column = 16, value = ('Carbon dioxide'))


for i in range(mexp):
    ws5.cell(row = 3+i, column = 1, value = (i+1))
    ws5.cell(row = 3+i, column = 2, value = (pred_cov1[:,0][i]))
    ws5.cell(row = 3+i, column = 3, value = (pred_cov1[:,1][i]))
    ws5.cell(row = 3+i, column = 4, value = (pred_cov1[:,2][i]))
    
    ws5.cell(row = 3+i, column = 7, value = (i+1))
    ws5.cell(row = 3+i, column = 8, value = (pred_cov2[:,0][i]))
    ws5.cell(row = 3+i, column = 9, value = (pred_cov2[:,1][i]))
    ws5.cell(row = 3+i, column = 10, value = (pred_cov2[:,2][i]))
    
    ws5.cell(row = 3+i, column = 13, value = (i+1))
    ws5.cell(row = 3+i, column = 14, value = (pred_cov3[:,0][i]))
    ws5.cell(row = 3+i, column = 15, value = (pred_cov3[:,1][i]))
    ws5.cell(row = 3+i, column = 16, value = (pred_cov3[:,2][i]))

ws6 = wb.create_sheet('distribution m2')
for i in range(mexp):
    ws6.cell(row = 1, column = i*4+1, value = ('%s %d' % ('y1', i+1)))
    ws6.cell(row = 1, column = i*4+2, value = ('%s %d' % ('y2', i+1)))
    ws6.cell(row = 1, column = i*4+3, value = ('%s %d' % ('y3', i+1)))

for i in range(mexp):
    for j in range(len(plot2[i])):
        ws6.cell(row = j+2, column = i*4+1, value = (plot2[i][j,0]))
        ws6.cell(row = j+2, column = i*4+2, value = (plot2[i][j,1]))
        ws6.cell(row = j+2, column = i*4+3, value = (plot2[i][j,2]))


ws7 = wb.create_sheet('distribution m3')
for i in range(mexp):
    ws7.cell(row = 1, column = i*4+1, value = ('%s %d' % ('y1', i+1)))
    ws7.cell(row = 1, column = i*4+2, value = ('%s %d' % ('y2', i+1)))
    ws7.cell(row = 1, column = i*4+3, value = ('%s %d' % ('y3', i+1)))

for i in range(mexp):
    for j in range(len(plot2[i])):
        ws7.cell(row = j+2, column = i*4+1, value = (plot3[i][j,0]))
        ws7.cell(row = j+2, column = i*4+2, value = (plot3[i][j,1]))
        ws7.cell(row = j+2, column = i*4+3, value = (plot3[i][j,2]))
        
ws8 = wb.create_sheet('distribution m2_2')
for i in range(mexp):
    ws8.cell(row = 1, column = i*4+1, value = ('%s %d' % ('y1', i+1)))
    ws8.cell(row = 1, column = i*4+2, value = ('%s %d' % ('y2', i+1)))
    ws8.cell(row = 1, column = i*4+3, value = ('%s %d' % ('y3', i+1)))

for i in range(mexp):
    for j in range(len(plot5[i])):
        ws8.cell(row = j+2, column = i*4+1, value = (plot5[i][j,0]))
        ws8.cell(row = j+2, column = i*4+2, value = (plot5[i][j,1]))
        ws8.cell(row = j+2, column = i*4+3, value = (plot5[i][j,2]))


ws9 = wb.create_sheet('distribution m3_2')
for i in range(mexp):
    ws9.cell(row = 1, column = i*4+1, value = ('%s %d' % ('y1', i+1)))
    ws9.cell(row = 1, column = i*4+2, value = ('%s %d' % ('y2', i+1)))
    ws9.cell(row = 1, column = i*4+3, value = ('%s %d' % ('y3', i+1)))

for i in range(mexp):
    for j in range(len(plot4[i])):
        ws9.cell(row = j+2, column = i*4+1, value = (plot4[i][j,0]))
        ws9.cell(row = j+2, column = i*4+2, value = (plot4[i][j,1]))
        ws9.cell(row = j+2, column = i*4+3, value = (plot4[i][j,2]))
wb.save('CH4oxidretronew_newresults_factorialplus8.xlsx')


# n_dexp = 1

# # MBDoE for model discrimination (between model 2 and 3)
# ig_md1, b_md1 = initialisation1_mbdoemd_BF(y_meas,km2,km3,est_m2,est_m3,FIM2_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_u)#,prng)
# ig_md0, b_md0 = initialisation0_mbdoemd_BF(y_meas,km2,km3,est_m2,est_m3,FIM2_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_u,prng)
# sol_md1 = minimize(mbdoemd_BF,ig_md1,method = 'SLSQP', bounds = (b_md1), args = (y_meas,km2,km3,est_m2,est_m3,FIM2_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_phi))
# sol_md0 = minimize(mbdoemd_BF,ig_md0,method = 'SLSQP', bounds = (b_md0), args = (y_meas,km2,km3,est_m2,est_m3,FIM2_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_phi))

# # MBDoE for model discrimination (between model 2 and 3)
# ig_md11, b_md11 = initialisation1_mbdoemd_BF(y_meas,km2r,km3,est_m2r,est_m3,FIM2r_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_u)#,prng)
# ig_md01, b_md01 = initialisation0_mbdoemd_BF(y_meas,km2r,km3,est_m2r,est_m3,FIM2r_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_u,prng)
# sol_md11 = minimize(mbdoemd_BF,ig_md11,method = 'SLSQP', bounds = (b_md11), args = (y_meas,km2r,km3,est_m2r,est_m3,FIM2r_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_phi))
# sol_md01 = minimize(mbdoemd_BF,ig_md01,method = 'SLSQP', bounds = (b_md01), args = (y_meas,km2r,km3,est_m2r,est_m3,FIM2r_obs[-1],FIM3_obs[-1],Pi[-1],n_dexp,n_phi))

# # MBDoE for parameter precision of model 3
# ig_pp1, b_pp1 = initialisation1_mbdoepp(y_meas,km3,est_m3,FIM3_obs[-1],Pi[-1],n_dexp,n_u)#,prng)
# ig_pp0, b_pp0 = initialisation0_mbdoepp(y_meas,km3,est_m3,FIM3_obs[-1],Pi[-1],n_dexp,n_u,prng)
# sol_pp1 = minimize(mbdoepp, ig_pp1, method = 'SLSQP', bounds = (b_pp1), args = (y_meas,km3,est_m3,FIM3_obs[-1],Pi[-1],n_dexp,n_phi))
# sol_pp0 = minimize(mbdoepp, ig_pp0, method = 'SLSQP', bounds = (b_pp0), args = (y_meas,km3,est_m3,FIM3_obs[-1],Pi[-1],n_dexp,n_phi))    
    
