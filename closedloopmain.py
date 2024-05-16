# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:53:08 2020

@author: p_aru
"""

import numpy as np
from scipy.integrate import odeint
import math
import scipy.stats as stats
from pyDOE2 import lhs, fullfact

import pyomo.environ as py
import pyomo.dae as pyd

def pdrop(x,L,u,As,Dp,eps,mu):
    """
    A function to describe pressure drop in the reactor

    Parameters
    ----------
    x : list[float]
        Differential state variable.
    L : float
        Reactor bed length.
    u : list[float]
        Process conditions.
    As : TYPE
        DESCRIPTION.
    Dp : TYPE
        DESCRIPTION.
    eps : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.

    Returns
    -------
    dPdz : TYPE
        DESCRIPTION.

    """
    T0, P0, rho0 = 20.0, 1.0, 1.2
    P = x[0]
    dPdz = -1.0*(((150.0*mu*(u[1]*(1e-6/60.0)*(P0*1e5/P)*((u[0]+273.15)/(T0+273.15)))*((1-eps)**2))/((Dp**2)*As*(eps**3))) + 
                ((1.75*(rho0*((u[1]*(1e-6/60.0))**2)*(P0*1e5/P)*((u[0]+273.15)/(T0+273.15)))*(1-eps))/(Dp*(As**2)*(eps**3))))
    return dPdz

def MLE_pd(c_theta,u_p,Pin_meas,sigma_P):
    """
    Maximum likelihood estimation (MLE) objective function to estimate c_theta

    Parameters
    ----------
    c_theta : TYPE
        DESCRIPTION.
    u_p : TYPE
        DESCRIPTION.
    Pin_meas : TYPE
        DESCRIPTION.
    sigma_P : TYPE
        DESCRIPTION.

    Returns
    -------
    MLE : TYPE
        DESCRIPTION.

    """
    T0, P0 = 20.0, 1.0
    As, Dp, eps, mu, L = 840e-9, 69e-6 , 0.40, 2.93e-5, 0.015
    n_exp = np.shape(u_p)[0]
    Pin = []
    Pin_c = []
    MLE = 0
    for i in range(n_exp):
        Pin += [odeint(pdrop,u_p[i][4]*1e5,np.linspace(L,0.0,5), args=(u_p[i],As,Dp,eps,mu))]
    for i in range(n_exp):
        Pin_c += [Pin[i][-1]*1e-5 + c_theta * (u_p[i][1]*(P0*1e5/Pin[i][-1])*((u_p[i][0]+273.15)/(T0+273.15)))]
    for i in range(n_exp):
        MLE += ((Pin_meas[i] - Pin_c[i])/sigma_P)**2
    return MLE

def Pinmodel(u,c_theta_hat):
    """
    Function to obtain inlet pressure by simulating pressure drop model
    at MLE

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    c_theta_hat : TYPE
        DESCRIPTION.

    Returns
    -------
    Pin_c_hat : TYPE
        DESCRIPTION.

    """
    T0, P0 = 20.0, 1.0
    As, Dp, eps, mu, L = 840e-9, 69e-6 , 0.40, 2.93e-5, 0.015
    x0 = u[4]*1e5
    t = np.linspace(L,0.0,5)
    Pin = odeint(pdrop,x0,t,args=(u,As,Dp,eps,mu))
    Pin_c_hat = Pin[-1]*1e-5 + c_theta_hat * (u[1]*(P0*1e5/Pin[-1])*((u[0]+273.15)/(T0+273.15)))
    return Pin_c_hat

def PinmodelErgun(u,c_theta_hat):
    As, Dp, eps, mu, L = 840e-9, 69e-6 , 0.40, 2.93e-5, 0.015
    x0 = u[4]*1e5
    t = np.linspace(L,0.0,5)
    Pin = odeint(pdrop,x0,t,args=(u,As,Dp,eps,mu))
    Pin_c_hat = Pin[-1]*1e-5
    return Pin_c_hat

def km1(x,t,u,theta,Pavg):
    """
    Power law model

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    Pavg : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    T0, P0 = 20.0, 1.0
    R, Tref = 8.314, 320
    yCH4 = x[0]
    yO2 = x[1]
    yCO2 = x[2]
    yH2O = x[3]
    k1 = math.exp(-theta[0] - (theta[1]*1e4/R) * ((1/(u[0]+273.15))-(1/(Tref+273.15))))
    r1 = k1 * Pavg * yCH4
    cf = ((R*(u[0]+273.15))/(Pavg*1e5*(u[1]*(1e-6/60)*(P0/Pavg)*((u[0]+273.15)/(T0+273.15)))))
    dyCH4dw = (-r1) * cf
    dyO2dw = 2.0*(-r1) * cf
    dyCO2dw = (r1) * cf
    dyH2Odw = 2.0*(r1) * cf
    return [dyCH4dw, dyO2dw, dyCO2dw, dyH2Odw]
        
def km2(x,t,u,theta,Pavg):
    """
    LHHW dissociative

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    Pavg : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    T0, P0 = 20.0, 1.0
    R, Tref = 8.314, 320
    yCH4 = x[0]
    yO2 = x[1]
    yCO2 = x[2]
    yH2O = x[3]
    k1 = math.exp(-theta[0] - (theta[1]*1e4/R) * ((1/(u[0]+273.15))-(1/(Tref+273.15))))
    k2 = math.exp(theta[2] - (-theta[3]*1e4/R) * ((1/(u[0]+273.15))-(1/(Tref+273.15))))
    k3 = math.exp(theta[4] - (-theta[5]*1e4/R) * ((1/(u[0]+273.15))-(1/(Tref+273.15))))
    r1 = (k1 * k3 * (Pavg * yCH4) * ((k2 * Pavg * yO2)**0.5))/((1 + (k3*Pavg*yCH4) + ((k2*Pavg*yO2)**0.5))**2)
    cf = ((R*(u[0]+273.15))/(Pavg*1e5*(u[1]*(1e-6/60)*(P0/Pavg)*((u[0]+273.15)/(T0+273.15)))))
    dyCH4dw = (-r1) * cf
    dyO2dw = 2.0*(-r1) * cf
    dyCO2dw = (r1) * cf
    dyH2Odw = 2.0*(r1) * cf
    return [dyCH4dw, dyO2dw, dyCO2dw, dyH2Odw]


def km3(x,t,u,theta,Pavg):
    """
    MVK molecular

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    Pavg : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    T0, P0 = 20.0, 1.0
    R, Tref = 8.314, 320
    yCH4 = x[0]
    yO2 = x[1]
    yCO2 = x[2]
    yH2O = x[3]
    k1 = math.exp(-theta[0] - (theta[1]*1e4/R) * ((1/(u[0]+273.15))-(1/(Tref+273.15))))
    k2 = math.exp(-theta[2] - (theta[3]*1e4/R) * ((1/(u[0]+273.15))-(1/(Tref+273.15))))
    k3 = math.exp(-theta[4] - (theta[5]*1e4/R) * ((1/(u[0]+273.15))-(1/(Tref+273.15))))
    r1 = (k1 * k2 * (Pavg**2) * yCH4 * yO2)/((k1*Pavg*yO2) + (2*k2*Pavg*yCH4) + ((k1*k2/k3)*(Pavg**2)*yCH4*yO2))
    cf = ((R*(u[0]+273.15))/(Pavg*1e5*(u[1]*(1e-6/60)*(P0/Pavg)*((u[0]+273.15)/(T0+273.15)))))
    dyCH4dw = (-r1) * cf
    dyO2dw = 2.0*(-r1) * cf
    dyCO2dw = (r1) * cf
    dyH2Odw = 2.0*(r1) * cf
    return [dyCH4dw, dyO2dw, dyCO2dw, dyH2Odw]

'''
General parameter estimation (regardless of the model)
'''

def mle_fun(theta,u,y,model,pdtheta):
    """
    MLE objective function to estimate kinetic model parameters

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.

    Returns
    -------
    MLE : TYPE
        DESCRIPTION.

    """
    mc = 0.01
    sigma = [0.00043, 0.00202, 0.00051]
    n_ymeas = np.shape(y)[1]
    y_hat = []
    MLE = 0
    for i in range(np.shape(u)[0]):
        x0 = [u[i][2],u[i][2]*u[i][3],0.0,0.0000000001]
        t = np.linspace(0.0,mc,5)
        y_hat += [odeint(model,x0,t,args=(u[i],theta,(Pinmodel(u[i],pdtheta)[0] + u[i][4])/2))[-1][:n_ymeas]]
    for i in range(np.shape(u)[0]):
        for j in range(n_ymeas):
            MLE += ((y[i,j] - y_hat[i][j])/sigma[j])**2
    return MLE


def insilicodatagenerator(truetheta,u,y,model,truepdtheta):
    mc = 0.01
    sigma = [0.00043, 0.00202, 0.00051]
    n_ymeas = np.shape(sigma)[0]
    x0 = [u[2],u[2]*u[3],0.0,0.0000000001]
    t = np.linspace(0.0,mc,5)
    y_hat = odeint(model,x0,t,args=(u,truetheta,(Pinmodel(u,truepdtheta)[0] + u[4])/2))[-1][:n_ymeas]
    return y_hat

def insilico_exp(u,truetheta,model,truepdtheta,y_cov):
    mc = 0.01
    x0 = [u[2],u[2]*u[3],0.0,0.0000000001]
    t = np.linspace(0.0,mc,5)
    yhat = odeint(model,x0,t,args=(u,truetheta,(Pinmodel(u,truepdtheta)[0] + u[4])/2))[-1][:np.shape(y_cov)[0]]
    error = np.random.multivariate_normal(np.array([0] * np.shape(y_cov)[0]), y_cov, 1)
    y = yhat + error
    return y[0]

def residual(u,theta,model,y_meas,pdtheta):
    mc = 0.01
    y_hat = []
    for i in range(np.shape(u)[0]):
        x0 = [u[i][2],u[i][2]*u[i][3],0.0,0.0000000001]
        t = np.linspace(0.0,mc,5)
        y_hat += [odeint(model,x0,t,args=(u[i],theta,(Pinmodel(u[i],pdtheta)[0] + u[i][4])/2))[-1][:np.shape(y_meas)[1]]]
    resid = y_meas - y_hat
    n_residual = np.zeros_like(resid)
    yhat = np.zeros_like(resid)
    n_residual[:,0] = resid[:,0] / 0.00043
    n_residual[:,1] = resid[:,1] / 0.00202
    n_residual[:,2] = resid[:,2] / 0.00051
    
    for i in range(np.shape(yhat)[0]):
        yhat[i] = y_hat[i]
#    return n_residual, yhat
    return resid, n_residual, yhat

def sen_fun(u,y,model,theta,pdtheta):
    """
    Function to calculate steady state sensitivities
    based on finite difference

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    epsilon = 0.001
    mc = 0.01
    p_matrix = np.zeros([np.shape(theta)[0]+1,np.shape(theta)[0]])
    for i in range(np.shape(theta)[0]+1):
        p_matrix[i] = theta
    for i in range(np.shape(theta)[0]):
        p_matrix[i][i] = theta[i] * (1 + epsilon)
    y_sen = []
    for theta in p_matrix:
        x0 = [u[2],u[2]*u[3],0.0,0.0000000001]
        t = np.linspace(0.0,mc,5)
        y_sen += [odeint(model,x0,t,args=(u,theta,(Pinmodel(u,pdtheta)[0] + u[4])/2))[-1][:np.shape(y)[1]]]
    s_matrix = np.zeros([np.shape(theta)[0],np.shape(y)[1]])
    for i in range(np.shape(theta)[0]):
        s_matrix[i] = (y_sen[i] - y_sen[-1])/(epsilon * theta[i])
    return np.transpose(s_matrix)

def FIM_fun(u,y,model,theta,pdtheta):
    """
    A function to calculate Fisher information matrix
    from parameter sensitivities

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.

    Returns
    -------
    FIM_matrix : TYPE
        DESCRIPTION.

    """
    epsilon = 0.001
    sigma = [0.00043, 0.00202, 0.00051]
    mc = 0.01
    p_matrix = np.zeros([np.shape(theta)[0]+1,np.shape(theta)[0]])
    for i in range(np.shape(theta)[0]+1):
        p_matrix[i] = theta
    for i in range(np.shape(theta)[0]):
        p_matrix[i][i] = theta[i] * (1 + epsilon)
    y_sen = []
    for theta in p_matrix:
        x0 = [u[2],u[2]*u[3],0.0,0.0000000001]
        t = np.linspace(0.0,mc,5)
        y_sen += [odeint(model,x0,t,args=(u,theta,(Pinmodel(u,pdtheta)[0] + u[4])/2))[-1][:3]]
    s_matrix = np.zeros([np.shape(theta)[0],np.shape(y)[1]])
    for i in range(np.shape(theta)[0]):
        s_matrix[i] = (y_sen[i] - y_sen[-1])/(epsilon * theta[i])
    FIM_matrix = np.zeros([np.shape(theta)[0],np.shape(theta)[0]])
    for k in range(np.shape(y)[1]):
        FIM_matrix = FIM_matrix + (1/(sigma[k]**2)) * np.outer(s_matrix[:,k],s_matrix[:,k])
    return FIM_matrix


def obs_FIM(up,y,model,pe,pdpe):
    """
    A function to calculate observed Fisher information matrix
    This corresponds to FIM computed based on measured 

    Parameters
    ----------
    up : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    pe : TYPE
        DESCRIPTION.
    pdpe : TYPE
        DESCRIPTION.

    Returns
    -------
    obsFIM : TYPE
        DESCRIPTION.

    """
    obsFIM = np.zeros([np.shape(pe)[0],np.shape(pe)[0]])
    for i in range(np.shape(up)[0]):
        obsFIM += FIM_fun(up[i],y,model,pe,pdpe)
    return obsFIM


def obs_COR(obscov):
    """
    A function to calculate observed correlation matrix

    Parameters
    ----------
    obscov : TYPE
        DESCRIPTION.

    Returns
    -------
    obsCOR : TYPE
        DESCRIPTION.

    """
    obsCOR = np.zeros([np.shape(obscov)[0],np.shape(obscov)[0]])
    for i in range(np.shape(obscov)[0]):
        for j in range(np.shape(obscov)[0]):
            obsCOR[i,j] = obscov[i,j]/(np.sqrt(obscov[i,i] * obscov[j,j]))
    return obsCOR

def tvalue_fun(pe,obscov,dof):
    """
    A function to perform t test to evaluate statistical precision
    of kinetic model parameter estimates

    Parameters
    ----------
    pe : TYPE
        DESCRIPTION.
    obscov : TYPE
        DESCRIPTION.
    dof : TYPE
        DESCRIPTION.

    Returns
    -------
    conf_interval : TYPE
        DESCRIPTION.
    t_value : TYPE
        DESCRIPTION.

    """
    alpha = 0.05
    conf_level = 1.0 - alpha
    conf_interval = np.zeros(np.shape(obscov)[0])
    t_value = np.zeros(np.shape(obscov)[0])
    for j in range(np.shape(obscov)[0]):
        conf_interval[j] = np.sqrt(obscov[j,j]) * stats.t.ppf((1 - (alpha/2)), dof)
        t_value[j] = pe[j]/(conf_interval[j])
    return conf_interval, t_value
    
def mprob1(chisq,dof):
    """
    A function to calculate probability of model correctness
    from the p values of chi-square test

    Parameters
    ----------
    chisq : TYPE
        DESCRIPTION.
    dof : TYPE
        DESCRIPTION.

    Returns
    -------
    probability1 : TYPE
        DESCRIPTION.
    probability2 : TYPE
        DESCRIPTION.

    """
    n_models = np.shape(chisq)[0]
    p_value = []
    sf_value = []
    probability1 = []
    probability2 = []
    for i in range(n_models):
        p_value += [1 - stats.chi2.cdf(chisq[i], dof[i])]
        sf_value += [stats.distributions.chi2.sf(chisq[i], dof[i])]
    for i in range(n_models):
        probability1 += [p_value[i] * 100 / sum(p_value)]
        probability2 += [sf_value[i] * 100 / sum(sf_value)]
    return probability1, probability2

def mprob2(chisq,dof):
    """
    A function to calculate probability of model correctness
    using inverse chisquare values

    Parameters
    ----------
    chisq : TYPE
        DESCRIPTION.
    dof : TYPE
        DESCRIPTION.

    Returns
    -------
    probability : TYPE
        DESCRIPTION.

    """
    n_models = np.shape(chisq)[0]
    inv_chisq = []
    probability = []
    for i in range(n_models):
        inv_chisq += [1 / chisq[i]]
    for i in range(n_models):
        probability += [inv_chisq[i] * 100 / sum(inv_chisq)]
    return probability

def confidence_ellipsoid31(est,cov,significance):
    # TODO: Write a general function to obtain confidence ellipse
    # TODO: for the critical parameters pair
    """
    A function to obtain confidence ellipse for the kinetic model 
    parameter pair indexed (3,1), given the covariance of model parameters

    Parameters
    ----------
    est : TYPE
        DESCRIPTION.
    cov : TYPE
        DESCRIPTION.
    significance : TYPE
        DESCRIPTION.

    Returns
    -------
    x_coordinate : TYPE
        DESCRIPTION.
    y_coordinate : TYPE
        DESCRIPTION.

    """
    redcov = np.zeros([2,2])
    
    redcov[0,0] = cov[3,3]
    redcov[0,1] = cov[3,1]
    redcov[1,0] = cov[1,3]
    redcov[1,1] = cov[1,1]
    
    eigenvalues,eigenvectors = np.linalg.eig(redcov)
    a = np.sqrt(eigenvalues)[0]                 # major axis
    b = np.sqrt(eigenvalues)[1]                 # minor axis
    z = stats.norm.ppf(significance)                                                   
    X = np.linspace(-a*z, a*z, 200)
    P = np.zeros(shape=(2*(len(X)),2))
    x_coordinate = []
    y_coordinate = []
    for i in range(len(X)):
        P[i,0] = X[i]
        P[(i+len(X)),0] = X[-(i+1)]
        P[i,1] = (b*z) * np.sqrt(1 - ((P[i,0]/(a*z))**2))
        P[(i+len(X)),1] = -(b*z) * np.sqrt(1 - ((P[(i+len(X)),0]/(a*z))**2))
    for i in range(len(X)):
        P[i] = np.dot(eigenvectors,P[i]) + [est[3],est[1]]
        P[(i+len(X))] = np.dot(eigenvectors,P[(i+len(X))]) + [est[3],est[1]]
    for i in range(2*len(X)):
        x_coordinate.append(P[i,0])
        y_coordinate.append(P[i,1])
    return x_coordinate, y_coordinate


def confidence_ellipsoid32(est,cov,significance):
    # TODO: Write a general function to obtain confidence ellipse
    # TODO: for the critical parameters pair
    """
    A function to obtain confidence ellipse for the kinetic model 
    parameter pair indexed (3,2), given the covariance of model parameters

    Parameters
    ----------
    est : TYPE
        DESCRIPTION.
    cov : TYPE
        DESCRIPTION.
    significance : TYPE
        DESCRIPTION.

    Returns
    -------
    x_coordinate : TYPE
        DESCRIPTION.
    y_coordinate : TYPE
        DESCRIPTION.

    """
    redcov = np.zeros([2,2])
    
    redcov[0,0] = cov[3,3]
    redcov[0,1] = cov[3,2]
    redcov[1,0] = cov[2,3]
    redcov[1,1] = cov[2,2]
    
    eigenvalues,eigenvectors = np.linalg.eig(redcov)
    a = np.sqrt(eigenvalues)[0]                 # major axis
    b = np.sqrt(eigenvalues)[1]                 # minor axis
    z = stats.norm.ppf(significance)                                                   
    X = np.linspace(-a*z, a*z, 200)
    P = np.zeros(shape=(2*(len(X)),2))
    x_coordinate = []
    y_coordinate = []
    for i in range(len(X)):
        P[i,0] = X[i]
        P[(i+len(X)),0] = X[-(i+1)]
        P[i,1] = (b*z) * np.sqrt(1 - ((P[i,0]/(a*z))**2))
        P[(i+len(X)),1] = -(b*z) * np.sqrt(1 - ((P[(i+len(X)),0]/(a*z))**2))
    for i in range(len(X)):
        P[i] = np.dot(eigenvectors,P[i]) + [est[3],est[2]]
        P[(i+len(X))] = np.dot(eigenvectors,P[(i+len(X))]) + [est[3],est[2]]
    for i in range(2*len(X)):
        x_coordinate.append(P[i,0])
        y_coordinate.append(P[i,1])
    return x_coordinate, y_coordinate

def confidence_ellipsoid1(est,cov,significance):
    """
    A function to obtain confidence ellipse of a two parameter model
    given the covariance matrix of model parameters

    Parameters
    ----------
    est : TYPE
        DESCRIPTION.
    cov : TYPE
        DESCRIPTION.
    significance : TYPE
        DESCRIPTION.

    Returns
    -------
    x_coordinate : TYPE
        DESCRIPTION.
    y_coordinate : TYPE
        DESCRIPTION.

    """
    eigenvalues,eigenvectors = np.linalg.eig(cov)
    a = np.sqrt(eigenvalues)[0]                 # major axis
    b = np.sqrt(eigenvalues)[1]                 # minor axis
    z = stats.norm.ppf(significance)                                                   
    X = np.linspace(-a*z, a*z, 200)
    P = np.zeros(shape=(2*(len(X)),2))
    x_coordinate = []
    y_coordinate = []
    for i in range(len(X)):
        P[i,0] = X[i]
        P[(i+len(X)),0] = X[-(i+1)]
        P[i,1] = (b*z) * np.sqrt(1 - ((P[i,0]/(a*z))**2))
        P[(i+len(X)),1] = -(b*z) * np.sqrt(1 - ((P[(i+len(X)),0]/(a*z))**2))
    for i in range(len(X)):
        P[i] = np.dot(eigenvectors,P[i]) + [est[0],est[1]]
        P[(i+len(X))] = np.dot(eigenvectors,P[(i+len(X))]) + [est[0],est[1]]
    for i in range(2*len(X)):
        x_coordinate.append(P[i,0])
        y_coordinate.append(P[i,1])
    return x_coordinate, y_coordinate

def pred_error(u,y,model,estimate,cov,pdtheta,df):
    """
    A function to evaluate model prediction error
    given the covariance of model parameters

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    estimate : TYPE
        DESCRIPTION.
    cov : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    y_hat : TYPE
        DESCRIPTION.
    obs_perror : TYPE
        DESCRIPTION.

    """
    sigma = [0.00043, 0.00202, 0.00051]
    mcov = np.identity(np.shape(y)[1])
    for i in range(np.shape(y)[1]):
        mcov[i,i] = sigma[i]**2
    mc = 0.01
    n_ymeas = np.shape(y)[1]
    alpha = 0.05
    t = np.linspace(0.0,mc,5)
    y_hat = []
    
    x0 = [u[2],u[2]*u[3],0.0,0.0000000001]
    
    y_hat = odeint(model,x0,t,args=(u,estimate,(Pinmodel(u,pdtheta)[0] + u[4])/2))[-1][:n_ymeas]
        
    obs_sen = sen_fun(u, y, model, estimate, pdtheta)
    obs_pcov = np.matmul(np.matmul(obs_sen, cov), np.transpose(obs_sen)) #+ mcov
    obs_perror = np.zeros(np.shape(obs_pcov)[0])
    for i in range(np.shape(obs_pcov)[0]):
        obs_perror[i] = np.sqrt(obs_pcov[i,i]) * stats.t.ppf((1 - (alpha/2)), df)
    return y_hat, obs_perror


def distributionplot1(u,estimate,model,pdtheta,errorbar):
    """
    A function to obtain prediction density plots showing the uncertainty of
    model predictions (based on sampling)

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    estimate : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    errorbar : TYPE
        DESCRIPTION.

    Returns
    -------
    yhat : TYPE
        DESCRIPTION.

    """
    mc = 0.01
    y_hat = []
    est1_b = np.linspace(estimate[0] - errorbar[0],estimate[0] + errorbar[0],num=5)
    est2_b = np.linspace(estimate[1] - errorbar[1],estimate[1] + errorbar[1],num=5)
    est3_b = np.linspace(estimate[2] - errorbar[2],estimate[2] + errorbar[2],num=5)
    est4_b = np.linspace(estimate[3] - errorbar[3],estimate[3] + errorbar[3],num=5)
    est5_b = np.linspace(estimate[4] - errorbar[4],estimate[4] + errorbar[4],num=5)
    est6_b = np.linspace(estimate[5] - errorbar[5],estimate[5] + errorbar[5],num=5)
    
    est1array, est2array, est3array, est4array, est5array, est6array = np.meshgrid(est1_b,est2_b,est3_b,est4_b,est5_b,est6_b)
    est1vals = np.ravel(est1array, order = 'F')
    est2vals = np.ravel(est2array, order = 'F')
    est3vals = np.ravel(est3array, order = 'F')
    est4vals = np.ravel(est4array, order = 'F')
    est5vals = np.ravel(est5array, order = 'F')
    est6vals = np.ravel(est6array, order = 'F')
    grid_data = np.zeros([len(est1vals),6])
    for i in range(len(est1vals)):
        grid_data[i,0] = est1vals[i]
        grid_data[i,1] = est2vals[i]
        grid_data[i,2] = est3vals[i]
        grid_data[i,3] = est4vals[i]
        grid_data[i,4] = est5vals[i]
        grid_data[i,5] = est6vals[i]
    for i in range(np.shape(grid_data)[0]):
        x0 = [u[2],u[2]*u[3],0.0,0.0000000001]
        t = np.linspace(0.0,mc,5)
        y_hat += [odeint(model,x0,t,args=(u,grid_data[i],(Pinmodel(u,pdtheta)[0] + u[4])/2))[-1][:3]]
    yhat = np.zeros_like(y_hat)
    
    for i in range(np.shape(yhat)[0]):
        yhat[i] = y_hat[i]
    return yhat

def distributionplot2(u,y,model,estimate,cov,pdtheta):
    """
    A function to obtain prediction density plots showing the uncertainty of
    model predictions (based on a gaussian prior on model predictions)

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    estimate : TYPE
        DESCRIPTION.
    cov : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.

    Returns
    -------
    error : TYPE
        DESCRIPTION.

    """
    sigma = [0.00043, 0.00202, 0.00051]
    mcov = np.identity(np.shape(y)[1])
    for i in range(np.shape(y)[1]):
        mcov[i,i] = sigma[i]**2
    mc = 0.01
    n_ymeas = np.shape(y)[1]
    alpha = 0.05
    t = np.linspace(0.0,mc,5)
    y_hat = []
    
    x0 = [u[2],u[2]*u[3],0.0,0.0000000001]
    
    y_hat = odeint(model,x0,t,args=(u,estimate,(Pinmodel(u,pdtheta)[0] + u[4])/2))[-1][:n_ymeas]
        
    obs_sen = sen_fun(u, y, model, estimate, pdtheta)
    obs_pcov = np.matmul(np.matmul(obs_sen, cov), np.transpose(obs_sen)) #+ mcov
    error = np.random.multivariate_normal(y_hat, obs_pcov, 500)
    return error

def distributionplot3(u,estimate,model,pdtheta,cov):
    """
    A function to obtain prediction density plots showing the uncertainty of
    model predictions (based on a gaussian prior on model parameters)

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    estimate : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    cov : TYPE
        DESCRIPTION.

    Returns
    -------
    yhat : TYPE
        DESCRIPTION.

    """
    mc = 0.01
    y_hat = []
    thetasample = np.random.multivariate_normal(estimate,cov,500)
    for i in range(np.shape(thetasample)[0]):
        x0 = [u[2],u[2]*u[3],0.0,0.0000000001]
        t = np.linspace(0.0,mc,5)
        y_hat += [odeint(model,x0,t,args=(u,thetasample[i],(Pinmodel(u,pdtheta)[0] + u[4])/2))[-1][:3]]
    yhat = np.zeros_like(y_hat)
    
    for i in range(np.shape(yhat)[0]):
        yhat[i] = y_hat[i]
    return yhat

def initialisation0_mbdoemd_BF(y,m2,m3,pe2,pe3,p2,p3,pdtheta,n_dexp,n_u,prng):
    """
    A function to find a good starting point for MBDoE for model discrimination
    based on LHS sampling

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    m2 : TYPE
        DESCRIPTION.
    m3 : TYPE
        DESCRIPTION.
    pe2 : TYPE
        DESCRIPTION.
    pe3 : TYPE
        DESCRIPTION.
    p2 : TYPE
        DESCRIPTION.
    p3 : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    n_dexp : TYPE
        DESCRIPTION.
    n_u : TYPE
        DESCRIPTION.
    prng : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    sigma = [0.00043, 0.00202, 0.00051]
    mcov = np.identity(np.shape(y)[1])
    for i in range(np.shape(y)[1]):
        mcov[i,i] = sigma[i]**2
    mc = 0.01
    control_bounds = [[250.0,350.0],[20.0,30.0],[0.005,0.025],[2.0,4.0],[1.3,1.3]]
    n_samples = 500
    n_resamples = 250
    lhs_points = lhs(np.shape(control_bounds)[0], samples = n_samples, criterion = 'maximin', iterations = 50, random_state = prng)
    lhs_actual = np.zeros([n_samples,n_u])
    for i in range(n_samples):
        lhs_actual[i,0] = lhs_points[i,0] * (350 - 250) + 250.0
        lhs_actual[i,1] = lhs_points[i,1] * (30 - 20) + 20.0
        lhs_actual[i,2] = lhs_points[i,2] * (0.025 - 0.005) + 0.005
        lhs_actual[i,3] = lhs_points[i,3] * (4 - 2) + 2.0
        lhs_actual[i,4] = 1.3
    
    actual_exp_seq = np.zeros([n_resamples,n_dexp,n_u])
    for i in range(n_resamples):
        actual_exp_seq[i] = lhs_actual[np.random.randint(n_samples, size = n_dexp),:]
    
    n_ymeas = np.shape(y)[1]
    y2_sample = []
    y3_sample = []
    for j in range(n_samples):
        x0 = [lhs_actual[j][2],lhs_actual[j][2]*lhs_actual[j][3],0.0,0.0000000001]
        t = np.linspace(0.0,mc,5)
        y2_sample += [odeint(m2,x0,t,args=(lhs_actual[j],pe2,(Pinmodel(lhs_actual[j],pdtheta)[0] + lhs_actual[j][4])/2))[-1][:n_ymeas]]
        y3_sample += [odeint(m3,x0,t,args=(lhs_actual[j],pe3,(Pinmodel(lhs_actual[j],pdtheta)[0] + lhs_actual[j][4])/2))[-1][:n_ymeas]]
    
    sen2_sample = []
    FIM2_sample = []
    pcov2_sample = []
    for j in range(n_samples):
        sen2_sample += [sen_fun(lhs_actual[j], y, m2, pe2, pdtheta)]
        FIM2_sample += [p2 + FIM_fun(lhs_actual[j], y, m2, pe2, pdtheta)]
    for j in range(n_samples):
        pcov2_sample += [np.matmul(np.matmul(sen2_sample[j], np.linalg.inv(FIM2_sample[j])), np.transpose(sen2_sample[j])) + mcov]

    sen3_sample = []
    FIM3_sample = []
    pcov3_sample = []
    for j in range(n_samples):
        sen3_sample += [sen_fun(lhs_actual[j], y, m3, pe3, pdtheta)]
        FIM3_sample += [p3 + FIM_fun(lhs_actual[j], y, m3, pe3, pdtheta)]
    for j in range(n_samples):
        pcov3_sample += [np.matmul(np.matmul(sen3_sample[j], np.linalg.inv(FIM3_sample[j])), np.transpose(sen3_sample[j])) + mcov]
        
    pcov23_sample = []
    for j in range(n_samples):
        pcov23_sample += [pcov2_sample[j] + pcov3_sample[j]]
        
    mdc23_sample = []
    for j in range(n_samples):
        mdc23_sample += [np.matmul(np.matmul(np.transpose(y2_sample[j] - y3_sample[j]), np.linalg.inv(pcov23_sample[j])), y2_sample[j] - y3_sample[j])]
    mdc23_sample_overall = np.zeros(1)
    for j in range(n_samples):
        mdc23_sample_overall += mdc23_sample[j]
    
    y2_resamples = []
    y3_resamples = []
    for j in range(n_resamples):
        x0 = [actual_exp_seq[j][0][2],actual_exp_seq[j][0][2]*actual_exp_seq[j][0][3],0.0,0.0000000001]
        t = np.linspace(0.0,mc,5)
        y2_resamples += [odeint(m2,x0,t,args=(actual_exp_seq[j][0],pe2,(Pinmodel(actual_exp_seq[j][0],pdtheta)[0] + actual_exp_seq[j][0][4])/2))[-1][:n_ymeas]]
        y3_resamples += [odeint(m3,x0,t,args=(actual_exp_seq[j][0],pe3,(Pinmodel(actual_exp_seq[j][0],pdtheta)[0] + actual_exp_seq[j][0][4])/2))[-1][:n_ymeas]]
        
    sen2_resamples = []
    FIM2_resamples = []
    pcov2_resamples = []
    for j in range(n_resamples):
        sen2_resamples += [sen_fun(actual_exp_seq[j][0],y,m2,pe2,pdtheta)]
        FIM2_resamples += [p2 + sum(FIM_fun(actual_exp_seq[j][k],y,m2,pe2,pdtheta) for k in range(n_dexp))]
    for j in range(n_resamples):
        pcov2_resamples += [np.matmul(np.matmul(sen2_resamples[j], np.linalg.inv(FIM2_resamples[j])), np.transpose(sen2_resamples[j])) + mcov]
    
    sen3_resamples = []
    FIM3_resamples = []
    pcov3_resamples = []
    for j in range(n_resamples):
        sen3_resamples += [sen_fun(actual_exp_seq[j][0],y,m3,pe3,pdtheta)]
        FIM3_resamples += [p3 + sum(FIM_fun(actual_exp_seq[j][k],y,m3,pe3,pdtheta) for k in range(n_dexp))]
    for j in range(n_resamples):
        pcov3_resamples += [np.matmul(np.matmul(sen3_resamples[j], np.linalg.inv(FIM3_resamples[j])), np.transpose(sen3_resamples[j])) + mcov]
        
    pcov23_resamples = []
    for j in range(n_resamples):
        pcov23_resamples += [pcov2_resamples[j] + pcov3_resamples[j]]
    
    mdc23_resamples = []
    for j in range(n_resamples):
        mdc23_resamples += [np.matmul(np.matmul(np.transpose(y2_resamples[j] - y3_resamples[j]), np.linalg.inv(pcov23_resamples[j])), y2_resamples[j] - y3_resamples[j])]
    
    rel_mdc = np.zeros(n_resamples)
    for j in range(n_resamples):
        rel_mdc[j] = (mdc23_resamples[j] / mdc23_sample_overall[0])

    lhs_ranking = actual_exp_seq[np.argmax(rel_mdc)]
    x0 = lhs_ranking[0]
    for i in range(1,n_dexp):
        x0 = np.append(x0,[lhs_ranking[i]])
    return x0[0:-1], control_bounds[0:-1] * n_dexp   

def initialisation1_mbdoemd_BF(y,m2,m3,pe2,pe3,p2,p3,pdtheta,n_dexp,n_u):
    """
    A function to find a good starting point for MBDoE for model discrimination
    based on full factorial sampling

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    m2 : TYPE
        DESCRIPTION.
    m3 : TYPE
        DESCRIPTION.
    pe2 : TYPE
        DESCRIPTION.
    pe3 : TYPE
        DESCRIPTION.
    p2 : TYPE
        DESCRIPTION.
    p3 : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    n_dexp : TYPE
        DESCRIPTION.
    n_u : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sigma = [0.00043, 0.00202, 0.00051]
    mcov = np.identity(np.shape(y)[1])
    for i in range(np.shape(y)[1]):
        mcov[i,i] = sigma[i]**2
    mc = 0.01
    control_bounds = [[250.0,350.0],[20.0,30.0],[0.005,0.025],[2.0,4.0],[1.3,1.3]]
    n_levels = 4
    ff_points = fullfact([n_levels,n_levels,n_levels,n_levels])
    n_samples = np.shape(ff_points)[0]
    ff_actual = np.zeros([np.shape(ff_points)[0],n_u])
    for i in range(np.shape(ff_points)[0]):
        ff_actual[i,0] = ff_points[i,0] * ((350 - 250)/(n_levels-1)) + 250.0
        ff_actual[i,1] = ff_points[i,1] * ((30 - 20)/(n_levels-1)) + 20.0
        ff_actual[i,2] = ff_points[i,2] * ((0.025 - 0.005)/(n_levels-1)) + 0.005
        ff_actual[i,3] = ff_points[i,3] * ((4 - 2)/(n_levels-1)) + 2.0
        ff_actual[i,4] = 1.3
    
    n_ymeas = np.shape(y)[1]
    y2_sample = []
    y3_sample = []
    for j in range(n_samples):
        x0 = [ff_actual[j][2],ff_actual[j][2]*ff_actual[j][3],0.0,0.0000000001]
        t = np.linspace(0.0,mc,5)
        y2_sample += [odeint(m2,x0,t,args=(ff_actual[j],pe2,(Pinmodel(ff_actual[j],pdtheta)[0] + ff_actual[j][4])/2))[-1][:n_ymeas]]
        y3_sample += [odeint(m3,x0,t,args=(ff_actual[j],pe3,(Pinmodel(ff_actual[j],pdtheta)[0] + ff_actual[j][4])/2))[-1][:n_ymeas]]
    
    sen2_sample = []
    FIM2_sample = []
    pcov2_sample = []
    for j in range(n_samples):
        sen2_sample += [sen_fun(ff_actual[j], y, m2, pe2, pdtheta)]
        FIM2_sample += [p2 + FIM_fun(ff_actual[j], y, m2, pe2, pdtheta)]
    for j in range(n_samples):
        pcov2_sample += [np.matmul(np.matmul(sen2_sample[j], np.linalg.inv(FIM2_sample[j])), np.transpose(sen2_sample[j])) + mcov]

    sen3_sample = []
    FIM3_sample = []
    pcov3_sample = []
    for j in range(n_samples):
        sen3_sample += [sen_fun(ff_actual[j], y, m3, pe3, pdtheta)]
        FIM3_sample += [p3 + FIM_fun(ff_actual[j], y, m3, pe3, pdtheta)]
    for j in range(n_samples):
        pcov3_sample += [np.matmul(np.matmul(sen3_sample[j], np.linalg.inv(FIM3_sample[j])), np.transpose(sen3_sample[j])) + mcov]
        
    pcov23_sample = []
    for j in range(n_samples):
        pcov23_sample += [pcov2_sample[j] + pcov3_sample[j]]
        
    mdc23_sample = []
    for j in range(n_samples):
        mdc23_sample += [np.matmul(np.matmul(np.transpose(y2_sample[j] - y3_sample[j]), np.linalg.inv(pcov23_sample[j])), (y2_sample[j] - y3_sample[j]))]

    ff_ranking = ff_actual[np.argmax(mdc23_sample)]
#    ff_ranking = ff_actual[np.argmin(-1.0 * np.array(mdc23_sample))]
    return ff_ranking[0:-1], control_bounds[0:-1] * n_dexp


def mbdoemd_BF(u,y,m2,m3,pe2,pe3,p2,p3,pdtheta,n_dexp,n_phi):
    # SEE: BuzziFerraris-Forzatti criterion
    """
    Objective function for MBDoE for model discrimination between two models

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    m2 : TYPE
        DESCRIPTION.
    m3 : TYPE
        DESCRIPTION.
    pe2 : TYPE
        DESCRIPTION.
    pe3 : TYPE
        DESCRIPTION.
    p2 : TYPE
        DESCRIPTION.
    p3 : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    n_dexp : TYPE
        DESCRIPTION.
    n_phi : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    sigma = [0.00043, 0.00202, 0.00051]
    mcov = np.identity(np.shape(y)[1])
    for i in range(np.shape(y)[1]):
        mcov[i,i] = sigma[i]**2
    mc = 0.01
    n_ymeas = np.shape(y)[1]
    y_hat_m2 = []
    y_hat_m3 = []
    for j in range(n_dexp):
        x0 = [u[j*n_phi:(j+1)*n_phi][2],u[j*n_phi:(j+1)*n_phi][2]*u[j*n_phi:(j+1)*n_phi][3],0.0,0.0000000001]
        t = np.linspace(0.0,mc,5)
        y_hat_m2 += [odeint(m2,x0,t,args=(np.hstack((u[j*n_phi:(j+1)*n_phi],np.array([1.27]))),pe2,(Pinmodel(np.hstack((u[j*n_phi:(j+1)*n_phi],np.array([1.27]))),pdtheta)[0] + 1.27)/2))[-1][:n_ymeas]]
        y_hat_m3 += [odeint(m3,x0,t,args=(np.hstack((u[j*n_phi:(j+1)*n_phi],np.array([1.27]))),pe3,(Pinmodel(np.hstack((u[j*n_phi:(j+1)*n_phi],np.array([1.27]))),pdtheta)[0] + 1.27)/2))[-1][:n_ymeas]]
    
    exp_sen2 = np.zeros([np.shape(y)[1],np.shape(pe2)[0]])
    for j in range(n_dexp):
        exp_sen2 += sen_fun(np.hstack((u[j*n_phi:(j+1)*n_phi],np.array([1.27]))), y, m2, pe2, pdtheta)
    exp_FIM2 = np.zeros([np.shape(pe2)[0],np.shape(pe2)[0]])
    exp_FIM2 += p2 # adding prior
    for j in range(n_dexp):
        exp_FIM2 += FIM_fun(np.hstack((u[j*n_phi:(j+1)*n_phi],np.array([1.27]))), y, m2, pe2, pdtheta)
    exp_cov2 = np.linalg.inv(exp_FIM2)
    exp_pcov2 = np.matmul(np.matmul(exp_sen2, exp_cov2), np.transpose(exp_sen2)) + mcov
    
    exp_sen3 = np.zeros([np.shape(y)[1],np.shape(pe3)[0]])
    for j in range(n_dexp):
        exp_sen3 += sen_fun(np.hstack((u[j*n_phi:(j+1)*n_phi],np.array([1.27]))), y, m3, pe3, pdtheta)
    exp_FIM3 = np.zeros([np.shape(pe3)[0],np.shape(pe3)[0]])
    exp_FIM3 += p3 # adding prior
    for j in range(n_dexp):
        exp_FIM3 += FIM_fun(np.hstack((u[j*n_phi:(j+1)*n_phi],np.array([1.27]))), y, m3, pe3, pdtheta)
    exp_cov3 = np.linalg.inv(exp_FIM3)
    exp_pcov3 = np.matmul(np.matmul(exp_sen3, exp_cov3), np.transpose(exp_sen3)) + mcov
    
    pcov23 = exp_pcov2 + exp_pcov3
    dc23 = np.matmul(np.matmul(np.transpose(y_hat_m2[0] - y_hat_m3[0]), np.linalg.inv(pcov23)), (y_hat_m2[0] - y_hat_m3[0]))
    
    return -1.0 * (dc23)

def initialisation0_mbdoepp(y,model,pe,prior,pdtheta,n_dexp,n_phi,prng):
    """
    A function to find good starting point for MBDoE for parameter precision
    based on LHS sampling

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    pe : TYPE
        DESCRIPTION.
    prior : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    n_dexp : TYPE
        DESCRIPTION.
    n_phi : TYPE
        DESCRIPTION.
    prng : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    control_bounds = [[250.0,350.0],[20.0,30.0],[0.005,0.025],[2.0,4.0],[1.3,1.3]]
    n_samples = 500
    n_resamples = 250
    lhs_points = lhs(4, samples = n_samples, criterion = 'maximin', iterations = 50, random_state = prng)
    lhs_actual = np.zeros([n_samples,n_phi])
    for i in range(n_samples):
        lhs_actual[i,0] = lhs_points[i,0] * (350 - 250) + 250.0
        lhs_actual[i,1] = lhs_points[i,1] * (30 - 20) + 20.0
        lhs_actual[i,2] = lhs_points[i,2] * (0.025 - 0.005) + 0.005
        lhs_actual[i,3] = lhs_points[i,3] * (4 - 2) + 2.0
        lhs_actual[i,4] = 1.3
    
    actual_exp_seq = np.zeros([n_resamples,n_dexp,n_phi])
    for i in range(n_resamples):
        actual_exp_seq[i] = lhs_actual[np.random.randint(n_samples, size = n_dexp),:]
    
    # for the full likelihood
    FIM_sample = np.zeros([np.shape(pe)[0],np.shape(pe)[0]])
    FIM_sample += prior # adding prior
    for j in range(n_samples):
        FIM_sample += FIM_fun(lhs_actual[j],y,model,pe,pdtheta)
 
    # for the full likelihood
    FIM_resamples = np.zeros([n_resamples,np.shape(pe)[0],np.shape(pe)[0]])
    for j in range(n_resamples):
        FIM_resamples[j] += prior # adding prior
    for j in range(n_resamples):
        FIM_resamples[j] += sum(FIM_fun(actual_exp_seq[j][k],y,model,pe,pdtheta) for k in range(n_dexp))
        
    # on FIM
    FIM_relative_det = np.zeros(n_resamples)
    FIM_relative_trace = np.zeros(n_resamples)
    FIM_relative_eigv = np.zeros(n_resamples)
    for i in range(n_resamples):
        FIM_relative_det[i] = 2 * sum(np.log(np.diag(np.linalg.cholesky(FIM_resamples[i])))) / 2 * sum(np.log(np.diag(np.linalg.cholesky(FIM_sample))))
        FIM_relative_trace[i] = np.trace(FIM_resamples[i]) / np.trace(FIM_sample)
        FIM_relative_eigv[i] = min(np.linalg.eigvals(FIM_resamples[i])) / min(np.linalg.eigvals(FIM_sample))
    lhs_ranking = actual_exp_seq[np.argmax(FIM_relative_det)]
    
    
    x0 = lhs_ranking[0]
    for i in range(1,n_dexp):
        x0 = np.append(x0,[lhs_ranking[i]])
        
    return x0[0:-1], control_bounds[0:-1] * n_dexp#, x01

def initialisation1_mbdoepp(y,model,pe,prior,pdtheta,n_dexp,n_u):
    """
    A function to find good starting point for MBDoE for parameter precision
    based on full factorial sampling

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    pe : TYPE
        DESCRIPTION.
    prior : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    n_dexp : TYPE
        DESCRIPTION.
    n_u : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    control_bounds = [[250.0,350.0],[20.0,30.0],[0.005,0.025],[2.0,4.0],[1.3,1.3]]
    n_levels = 8
    ff_points = fullfact([n_levels,n_levels,n_levels,n_levels])
    n_samples = np.shape(ff_points)[0]
    ff_actual = np.zeros([np.shape(ff_points)[0],n_u])
    for i in range(np.shape(ff_points)[0]):
        ff_actual[i,0] = ff_points[i,0] * ((350 - 250)/(n_levels-1)) + 250.0
        ff_actual[i,1] = ff_points[i,1] * ((30 - 20)/(n_levels-1)) + 20.0
        ff_actual[i,2] = ff_points[i,2] * ((0.025 - 0.005)/(n_levels-1)) + 0.005
        ff_actual[i,3] = ff_points[i,3] * ((4 - 2)/(n_levels-1)) + 2.0
        ff_actual[i,4] = 1.3
    
    # for the full likelihood
    FIM_sample = np.zeros([n_samples,np.shape(pe)[0],np.shape(pe)[0]])
    for j in range(n_samples):
        FIM_sample[j] += prior
    for j in range(n_samples):
        FIM_sample[j] += FIM_fun(ff_actual[j],y,model,pe,pdtheta)
        
    # on FIM
    FIM_det = np.zeros(n_samples)
    FIM_trace = np.zeros(n_samples)
    FIM_eigv = np.zeros(n_samples)
    for i in range(n_samples):
        FIM_det[i] = 2 * sum(np.log(np.diag(np.linalg.cholesky(FIM_sample[i]))))
        FIM_trace[i] = np.trace(FIM_sample[i])
        FIM_eigv[i] = min(np.linalg.eigvals(FIM_sample[i]))
    ff_ranking = ff_actual[np.argmax(FIM_det)]
#    x0 = ff_ranking[0]
#    for i in range(1,n_dexp):
#        x0 = np.append(x0,[ff_ranking[i]])
    return ff_ranking[0:-1], control_bounds[0:-1] * n_dexp



def mbdoepp(u,y,model,pe,prior,pdtheta,n_dexp,n_phi):
    """
    Objective function for MBDoE for parameter precision

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    pe : TYPE
        DESCRIPTION.
    prior : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    n_dexp : TYPE
        DESCRIPTION.
    n_phi : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    exp_FIM = np.zeros([np.shape(pe)[0],np.shape(pe)[0]])
    exp_FIM += prior # adding prior
    for j in range(n_dexp):
        #u[(j+1)*(n_phi-1)+j] = 1.3
        exp_FIM += FIM_fun(np.hstack((u[j*n_phi:(j+1)*n_phi],np.array([1.27]))), y, model, pe, pdtheta)
    exp_cov = np.linalg.inv(exp_FIM)
    # return -math.log(min(np.linalg.eigvals(exp_FIM)))
#    return math. log(np.trace(exp_cov))
#    return math.log(max(np.linalg.eigvals(exp_cov)))
    return 2 * sum(np.log(np.diag(np.linalg.cholesky(exp_cov))))

    

def tdomain(tarray):
    """
    A function to define the finite element points
    which include all sampling times

    Parameters
    ----------
    tarray : TYPE
        DESCRIPTION.

    Returns
    -------
    tk : TYPE
        DESCRIPTION.

    """
    td = []
    for i in range(np.shape(tarray)[0]):
        td += list(tarray[i])
    tk = sorted(np.unique(td))
    tk[0] = 0
    return tk

def xmeas(tarray, yarray):
    """
    A function to define the sampling times and measurements

    Parameters
    ----------
    tarray : TYPE
        DESCRIPTION.
    yarray : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    idx = list()
    for i in list(range(np.shape(tarray)[0])):
        for j in tarray[i]:
            idx.append((i,j))
    
    val_x1 = []
    val_x2 = []
    val_x3 = []
#    val_x4 = []
    for i in range(np.shape(yarray)[0]):
        val_x1 += list(yarray[i][:,0])
        val_x2 += list(yarray[i][:,1])
        val_x3 += list(yarray[i][:,2])
#        val_x4 += list(yarray[i][:,3])
        
    dic_x1 = {idx[i]:val_x1[i] for i in range(np.shape(val_x1)[0])}
    dic_x2 = {idx[i]:val_x2[i] for i in range(np.shape(val_x2)[0])}
    dic_x3 = {idx[i]:val_x3[i] for i in range(np.shape(val_x3)[0])}
#    dic_x4 = {idx[i]:val_x4[i] for i in range(np.shape(val_x4)[0])}
    return idx, dic_x1, dic_x2, dic_x3#, dic_x4

def controls(uarray):
    """
    A function to define the controls

    Parameters
    ----------
    uarray : TYPE
        DESCRIPTION.

    Returns
    -------
    idx : TYPE
        DESCRIPTION.
    dic : TYPE
        DESCRIPTION.

    """
    idx = list()
    for i in list(range(np.shape(uarray)[0])):
        for j in list(range(np.shape(uarray)[1])):
            idx.append((i,j))
            
    val = []
    for i in range(np.shape(uarray)[0]):
        val += list(uarray[i])
        
    dic = {idx[i]:val[i] for i in range(np.shape(val)[0])}
    return idx, dic

def pcontrols(uarray,ptheta):
    """
    A function to define the pressure controls

    Parameters
    ----------
    uarray : TYPE
        DESCRIPTION.
    ptheta : TYPE
        DESCRIPTION.

    Returns
    -------
    dic : TYPE
        DESCRIPTION.

    """
    idx = list(range(np.shape(uarray)[0]))
    val = []
    for i in range(np.shape(uarray)[0]):
        val += [Pinmodel(uarray[i],ptheta)[0]]
    dic = {idx[i]:val[i] for i in range(np.shape(val)[0])}
    return dic

def discdom(T, N, scheme_name):
    """
    A function for time discretization

    Parameters
    ----------
    T : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    scheme_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    model = py.ConcreteModel()
#    ft = int(T)
    ft = (T)
    total_elements = int(N)
    element_length = (ft/total_elements)
    
#    scheme_name = 'LAGRANGE-LEGENDRE'
#    scheme_name = 'LAGRANGE-RADAU'
    discretizer = py.TransformationFactory('dae.collocation')
    
    d = 4
    
    model.tau = pyd.ContinuousSet(bounds = (0,ft))
    discretizer.apply_to(model, wrt = model.tau, nfe = total_elements, ncp = d, scheme = scheme_name)
    
    return sorted(model.tau)

##### Pyomo models #####


def optikm1(u_array, y_array, t_array, pdtheta, ig, lb, ub):
    """
    Pyomo model for power law kinetics

    Parameters
    ----------
    u_array : TYPE
        DESCRIPTION.
    y_array : TYPE
        DESCRIPTION.
    t_array : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    ig : TYPE
        DESCRIPTION.
    lb : TYPE
        DESCRIPTION.
    ub : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    T0, P0 = 20.0, 1.0
    R, Tref = 8.314, 320
    y_idx, x1m, x2m, x3m = xmeas(t_array, y_array)
    tsp = tdomain(t_array)
    u_idx, up = controls(u_array)
    Pin = pcontrols(u_array, pdtheta)
    sigma = [0.00043, 0.00202, 0.00051]
    d = 4
#    scheme_name = 'LAGRANGE-LEGENDRE'
    scheme_name = 'LAGRANGE-RADAU'
    model = py.ConcreteModel()
    model.t = pyd.ContinuousSet(initialize = tsp)
    model.N = py.Set(initialize = list(range(np.shape(u_array)[0])))
    model.x1meas = py.Param(y_idx, initialize = x1m)
    model.x2meas = py.Param(y_idx, initialize = x2m)
    model.x3meas = py.Param(y_idx, initialize = x3m)
#    model.x4meas = py.Param(y_idx, initialize = x4m)
    model.u = py.Param(u_idx, initialize = up)
    model.P = py.Param(model.N, initialize = Pin)
    
    model.theta_idx = py.Set(initialize = list(range(np.shape(ig)[0])))
    theta_lb_dict = dict(list(enumerate(lb)))
    theta_ub_dict = dict(list(enumerate(ub)))
#    theta_dict = dict(list(enumerate([0.2,0.1])))
    theta_dict = dict(list(enumerate(ig)))

    def parmbounds(model,i):
        return (theta_lb_dict[i], theta_ub_dict[i])
    
    
    # declare differential variables
    model.x1 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    model.x2 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    model.x3 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    model.x4 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    
    # declare derivatives
    model.dx1dt = pyd.DerivativeVar(model.x1, wrt = model.t)
    model.dx2dt = pyd.DerivativeVar(model.x2, wrt = model.t)
    model.dx3dt = pyd.DerivativeVar(model.x3, wrt = model.t)
    model.dx4dt = pyd.DerivativeVar(model.x4, wrt = model.t)
    
    # declare model parameters
    model.theta = py.Var(model.theta_idx, initialize = theta_dict, bounds = parmbounds)
    
    def diffeqn1(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = k1 * (((model.P[n] + model.u[n,4])/2) * model.x1[n,t])
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2)) * \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx1dt[n,t] == -(r1) * cf
    model.x1cons = py.Constraint(model.N, model.t, rule = diffeqn1)
    
    def diffeqn2(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = k1 * (((model.P[n] + model.u[n,4])/2) * model.x1[n,t])
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2)) * \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx2dt[n,t] == -2 * (r1) * cf
    model.x2cons = py.Constraint(model.N, model.t, rule = diffeqn2)
    
    def diffeqn3(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = k1 * (((model.P[n] + model.u[n,4])/2) * model.x1[n,t])
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2)) * \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx3dt[n,t] == (r1) * cf
    model.x3cons = py.Constraint(model.N, model.t, rule = diffeqn3)
    
    def diffeqn4(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = k1 * (((model.P[n] + model.u[n,4])/2) * model.x1[n,t])
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2)) * \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx4dt[n,t] == 2 * (r1) * cf
    model.x4cons = py.Constraint(model.N, model.t, rule = diffeqn4)
    
    def init_conditions1(model,n):
        return model.x1[n,0] == py.value(model.u[n,2])
    def init_conditions2(model,n):
        return model.x2[n,0] == py.value(model.u[n,2]) * py.value(model.u[n,3])
    def init_conditions3(model,n):
        return model.x3[n,0] == 0.0
    def init_conditions4(model,n):
        return model.x4[n,0] == 0.0000000001
    
    model.init_cond1 = py.Constraint(model.N, rule = init_conditions1)
    model.init_cond2 = py.Constraint(model.N, rule = init_conditions2)
    model.init_cond3 = py.Constraint(model.N, rule = init_conditions3)
    model.init_cond4 = py.Constraint(model.N, rule = init_conditions4)
    
    discretizer = py.TransformationFactory('dae.collocation')
#    discretizer.apply_to(model, wrt = model.t, nfe = total_elements, ncp = 5)
    discretizer.apply_to(model, wrt = model.t, ncp = d, scheme = scheme_name)
    
    def obj_expression(model):  # To perform dynamic simulation at different parameter values (including the initial conditions), the trick is to put the initial conditions later in the for loop
        chisq_1 = sum((((model.x1meas[j] - model.x1[j])**2) / sigma[0]**2) + (((model.x2meas[j] - model.x2[j])**2) / sigma[1]**2) + (((model.x3meas[j] - model.x3[j])**2) / sigma[2]**2) for j in y_idx)
        obj_fun = chisq_1
        return obj_fun
    model.objfun = py.Objective(rule = obj_expression)
    
    return model


def optikm2(u_array, y_array, t_array, pdtheta, ig, lb, ub):
    """
    Pyomo model for LHHW dissociative kinetics

    Parameters
    ----------
    u_array : TYPE
        DESCRIPTION.
    y_array : TYPE
        DESCRIPTION.
    t_array : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    ig : TYPE
        DESCRIPTION.
    lb : TYPE
        DESCRIPTION.
    ub : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    T0, P0 = 20.0, 1.0
    R, Tref = 8.314, 320
    y_idx, x1m, x2m, x3m = xmeas(t_array, y_array)
    tsp = tdomain(t_array)
    u_idx, up = controls(u_array)
    Pin = pcontrols(u_array, pdtheta)
    sigma = [0.00043, 0.00202, 0.00051]
    d = 4
#    scheme_name = 'LAGRANGE-LEGENDRE'
    scheme_name = 'LAGRANGE-RADAU'
    
    model = py.ConcreteModel()
    model.t = pyd.ContinuousSet(initialize = tsp)
    model.N = py.Set(initialize = list(range(np.shape(u_array)[0])))
    model.x1meas = py.Param(y_idx, initialize = x1m)
    model.x2meas = py.Param(y_idx, initialize = x2m)
    model.x3meas = py.Param(y_idx, initialize = x3m)
#    model.x4meas = py.Param(y_idx, initialize = x4m)
    model.u = py.Param(u_idx, initialize = up)
    model.P = py.Param(model.N, initialize = Pin)
    
    model.theta_idx = py.Set(initialize = list(range(np.shape(ig)[0])))
    theta_lb_dict = dict(list(enumerate(lb)))
    theta_ub_dict = dict(list(enumerate(ub)))
#    theta_dict = dict(list(enumerate([0.2,0.1])))
    theta_dict = dict(list(enumerate(ig)))

    def parmbounds(model,i):
        return (theta_lb_dict[i], theta_ub_dict[i])
    
    
    # declare differential variables
    model.x1 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    model.x2 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    model.x3 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    model.x4 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    
    # declare derivatives
    model.dx1dt = pyd.DerivativeVar(model.x1, wrt = model.t)
    model.dx2dt = pyd.DerivativeVar(model.x2, wrt = model.t)
    model.dx3dt = pyd.DerivativeVar(model.x3, wrt = model.t)
    model.dx4dt = pyd.DerivativeVar(model.x4, wrt = model.t)
    
    # declare model parameters
    model.theta = py.Var(model.theta_idx, initialize = theta_dict, bounds = parmbounds)
    
    def diffeqn1(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k2 = (py.exp(model.theta[2] - (-model.theta[3] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k3 = (py.exp(model.theta[4] - (-model.theta[5] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = ((k1 * k3 * (((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) * \
               ((k2 * ((((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + 1e-60))**0.5)) / \
               (((1 + (k3 * ((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) + \
               ((k2 * ((((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + 1e-60))**0.5))**2)))
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2))* \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx1dt[n,t] == -(r1) * cf
    model.x1cons = py.Constraint(model.N, model.t, rule = diffeqn1)
    
    def diffeqn2(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k2 = (py.exp(model.theta[2] - (-model.theta[3] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k3 = (py.exp(model.theta[4] - (-model.theta[5] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = ((k1 * k3 * (((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) * \
               ((k2 * ((((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + 1e-60))**0.5)) / \
               (((1 + (k3 * ((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) + \
               ((k2 * ((((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + 1e-60))**0.5))**2)))
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2))* \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx2dt[n,t] == -2 * (r1) * cf
    model.x2cons = py.Constraint(model.N, model.t, rule = diffeqn2)
    
    def diffeqn3(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k2 = (py.exp(model.theta[2] - (-model.theta[3] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k3 = (py.exp(model.theta[4] - (-model.theta[5] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = ((k1 * k3 * (((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) * \
               ((k2 * ((((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + 1e-60))**0.5)) / \
               (((1 + (k3 * ((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) + \
               ((k2 * ((((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + 1e-60))**0.5))**2)))
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2))* \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx3dt[n,t] == (r1) * cf
    model.x3cons = py.Constraint(model.N, model.t, rule = diffeqn3)
    
    def diffeqn4(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k2 = (py.exp(model.theta[2] - (-model.theta[3] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k3 = (py.exp(model.theta[4] - (-model.theta[5] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = ((k1 * k3 * (((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) * \
               ((k2 * ((((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + 1e-60))**0.5)) / \
               (((1 + (k3 * ((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) + \
               ((k2 * ((((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + 1e-60))**0.5))**2)))
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2))* \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx4dt[n,t] == 2 * (r1) * cf
    model.x4cons = py.Constraint(model.N, model.t, rule = diffeqn4)
    
    def init_conditions1(model,n):
        return model.x1[n,0] == py.value(model.u[n,2])
    def init_conditions2(model,n):
        return model.x2[n,0] == py.value(model.u[n,2]) * py.value(model.u[n,3])
    def init_conditions3(model,n):
        return model.x3[n,0] == 0.0
    def init_conditions4(model,n):
        return model.x4[n,0] == 0.0000000001
    
    model.init_cond1 = py.Constraint(model.N, rule = init_conditions1)
    model.init_cond2 = py.Constraint(model.N, rule = init_conditions2)
    model.init_cond3 = py.Constraint(model.N, rule = init_conditions3)
    model.init_cond4 = py.Constraint(model.N, rule = init_conditions4)
    
    discretizer = py.TransformationFactory('dae.collocation')
#    discretizer.apply_to(model, wrt = model.t, nfe = total_elements, ncp = 5)
    discretizer.apply_to(model, wrt = model.t, ncp = d, scheme = scheme_name)
    
    def obj_expression(model):  # To perform dynamic simulation at different parameter values (including the initial conditions), the trick is to put the initial conditions later in the for loop
        chisq_1 = sum((((model.x1meas[j] - model.x1[j])**2) / sigma[0]**2) + (((model.x2meas[j] - model.x2[j])**2) / sigma[1]**2) + (((model.x3meas[j] - model.x3[j])**2) / sigma[2]**2) for j in y_idx)
        obj_fun = chisq_1
        return obj_fun
    model.objfun = py.Objective(rule = obj_expression)
    
    return model

def optikm3(u_array, y_array, t_array, pdtheta, ig, lb, ub):
    """
    Pyomo model for MVK molecular kinetics

    Parameters
    ----------
    u_array : TYPE
        DESCRIPTION.
    y_array : TYPE
        DESCRIPTION.
    t_array : TYPE
        DESCRIPTION.
    pdtheta : TYPE
        DESCRIPTION.
    ig : TYPE
        DESCRIPTION.
    lb : TYPE
        DESCRIPTION.
    ub : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    T0, P0 = 20.0, 1.0
    R, Tref = 8.314, 320
    y_idx, x1m, x2m, x3m = xmeas(t_array, y_array)
    tsp = tdomain(t_array)
    u_idx, up = controls(u_array)
    Pin = pcontrols(u_array, pdtheta)
    sigma = [0.00043, 0.00202, 0.00051]
    d = 4
#    scheme_name = 'LAGRANGE-LEGENDRE'
    scheme_name = 'LAGRANGE-RADAU'
    model = py.ConcreteModel()
    model.t = pyd.ContinuousSet(initialize = tsp)
    model.N = py.Set(initialize = list(range(np.shape(u_array)[0])))
    model.x1meas = py.Param(y_idx, initialize = x1m)
    model.x2meas = py.Param(y_idx, initialize = x2m)
    model.x3meas = py.Param(y_idx, initialize = x3m)
#    model.x4meas = py.Param(y_idx, initialize = x4m)
    model.u = py.Param(u_idx, initialize = up)
    model.P = py.Param(model.N, initialize = Pin)
    
    model.theta_idx = py.Set(initialize = list(range(np.shape(ig)[0])))
    theta_lb_dict = dict(list(enumerate(lb)))
    theta_ub_dict = dict(list(enumerate(ub)))
#    theta_dict = dict(list(enumerate([0.2,0.1])))
    theta_dict = dict(list(enumerate(ig)))

    def parmbounds(model,i):
        return (theta_lb_dict[i], theta_ub_dict[i])
    
    
    # declare differential variables
    model.x1 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    model.x2 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    model.x3 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    model.x4 = py.Var(model.N, model.t, within = py.NonNegativeReals, bounds = (0,1))
    
    # declare derivatives
    model.dx1dt = pyd.DerivativeVar(model.x1, wrt = model.t)
    model.dx2dt = pyd.DerivativeVar(model.x2, wrt = model.t)
    model.dx3dt = pyd.DerivativeVar(model.x3, wrt = model.t)
    model.dx4dt = pyd.DerivativeVar(model.x4, wrt = model.t)
    
    # declare model parameters
    model.theta = py.Var(model.theta_idx, initialize = theta_dict, bounds = parmbounds)
    
    def diffeqn1(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k2 = (py.exp(-model.theta[2] - (model.theta[3] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k3 = (py.exp(-model.theta[4] - (model.theta[5] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = ((k1 * k2 * (((model.P[n] + model.u[n,4])/2)**2) * model.x1[n,t] * model.x2[n,t]) / \
               (((k1 * ((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + \
               (2* k2 * ((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) + \
               ((k1 * k2/k3) * (((model.P[n] + model.u[n,4])/2)**2) * model.x1[n,t] * model.x2[n,t]))))
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2))* \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx1dt[n,t] == -(r1) * cf
    model.x1cons = py.Constraint(model.N, model.t, rule = diffeqn1)
    
    def diffeqn2(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k2 = (py.exp(-model.theta[2] - (model.theta[3] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k3 = (py.exp(-model.theta[4] - (model.theta[5] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = ((k1 * k2 * (((model.P[n] + model.u[n,4])/2)**2) * model.x1[n,t] * model.x2[n,t]) / \
               (((k1 * ((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + \
               (2* k2 * ((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) + \
               ((k1 * k2/k3) * (((model.P[n] + model.u[n,4])/2)**2) * model.x1[n,t] * model.x2[n,t]))))
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2))* \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx2dt[n,t] == -2 * (r1) * cf
    model.x2cons = py.Constraint(model.N, model.t, rule = diffeqn2)
    
    def diffeqn3(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k2 = (py.exp(-model.theta[2] - (model.theta[3] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k3 = (py.exp(-model.theta[4] - (model.theta[5] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = ((k1 * k2 * (((model.P[n] + model.u[n,4])/2)**2) * model.x1[n,t] * model.x2[n,t]) / \
               (((k1 * ((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + \
               (2* k2 * ((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) + \
               ((k1 * k2/k3) * (((model.P[n] + model.u[n,4])/2)**2) * model.x1[n,t] * model.x2[n,t]))))
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2))* \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx3dt[n,t] == (r1) * cf
    model.x3cons = py.Constraint(model.N, model.t, rule = diffeqn3)
    
    def diffeqn4(model,n,t):
        k1 = (py.exp(-model.theta[0] - (model.theta[1] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k2 = (py.exp(-model.theta[2] - (model.theta[3] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        k3 = (py.exp(-model.theta[4] - (model.theta[5] * 1e4/R) * \
               ((1/(model.u[n,0]+273.15))-(1/(Tref+273.15)))))
        r1 = ((k1 * k2 * (((model.P[n] + model.u[n,4])/2)**2) * model.x1[n,t] * model.x2[n,t]) / \
               (((k1 * ((model.P[n] + model.u[n,4])/2) * model.x2[n,t]) + \
               (2* k2 * ((model.P[n] + model.u[n,4])/2) * model.x1[n,t]) + \
               ((k1 * k2/k3) * (((model.P[n] + model.u[n,4])/2)**2) * model.x1[n,t] * model.x2[n,t]))))
        cf = ((R*(model.u[n,0]+273.15))/(((model.P[n] + model.u[n,4])/2) * \
               1e5 * (model.u[n,1]*(1e-6/60)*(P0/((model.P[n] + model.u[n,4])/2))* \
               ((model.u[n,0]+273.15)/(T0+273.15)))))
        if model.t == 0:
            return py.Constraint.Skip
        return model.dx4dt[n,t] == 2 * (r1) * cf
    model.x4cons = py.Constraint(model.N, model.t, rule = diffeqn4)
    

    
    def init_conditions1(model,n):
        return model.x1[n,0] == py.value(model.u[n,2])
    def init_conditions2(model,n):
        return model.x2[n,0] == py.value(model.u[n,2]) * py.value(model.u[n,3])
    def init_conditions3(model,n):
        return model.x3[n,0] == 0.0
    def init_conditions4(model,n):
        return model.x4[n,0] == 0.0000000001
    
    model.init_cond1 = py.Constraint(model.N, rule = init_conditions1)
    model.init_cond2 = py.Constraint(model.N, rule = init_conditions2)
    model.init_cond3 = py.Constraint(model.N, rule = init_conditions3)
    model.init_cond4 = py.Constraint(model.N, rule = init_conditions4)
    
    discretizer = py.TransformationFactory('dae.collocation')
#    discretizer.apply_to(model, wrt = model.t, nfe = total_elements, ncp = 5)
    discretizer.apply_to(model, wrt = model.t, ncp = d, scheme = scheme_name)
    
    def obj_expression(model):  # To perform dynamic simulation at different parameter values (including the initial conditions), the trick is to put the initial conditions later in the for loop
        chisq_1 = sum((((model.x1meas[j] - model.x1[j])**2) / sigma[0]**2) + (((model.x2meas[j] - model.x2[j])**2) / sigma[1]**2) + (((model.x3meas[j] - model.x3[j])**2) / sigma[2]**2) for j in y_idx)
        obj_fun = chisq_1
        return obj_fun
    model.objfun = py.Objective(rule = obj_expression)
    
    return model