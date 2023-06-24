#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 21:55:46 2022

@author: mariacm
"""
import numpy as np

def get_eig(TEns, V12):
    from numpy.linalg import eigh
    
    ham = np.array([[TEns[0],V12],[V12,TEns[1]]])
    evals, evecs = eigh(ham)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:,idx]
    return evals, evecs

def eig_osc(osc_st, evecs, evals):
    osc_eig1 = evecs[0,0]*osc_st[0] + evecs[1,0]*osc_st[1]
    osc_eig2 = evecs[0,1]*osc_st[0] + evecs[1,1]*osc_st[1]
    # print('**',osc_st[0]>osc_st[1] and  osc_eig1>osc_eig2)
    # print('**',evecs[:,0], evals[0], evals[1])
    return osc_eig1, osc_eig2

def dimer_abs(TEns1, TEns2, Vs, osts1, osts2, abs_fn, E, bwidth=None):
    npoints = len(TEns1)
    eig_sum = []
    for ti in range(npoints):
        Ten_idx = np.argsort([TEns1[ti], TEns2[ti]])
        Ten_site = np.array([TEns1[ti], TEns2[ti]])#[Ten_idx]
        os_site = np.array([osts1[ti], osts2[ti]])#[Ten_idx]
        evals, evecs = get_eig(Ten_site, Vs[ti])

        osc_eigA, osc_eigB = eig_osc(os_site, evecs, evals)
        abs_i = osc_eigA*abs_fn(evals[0],E,bwidth) + osc_eigB*abs_fn(evals[1],E,bwidth)
        eig_sum.append(abs_i)
    eig_sum = np.array(eig_sum)
    I_array = np.sum(eig_sum,axis=0)
    
    return abs(I_array)

def delta_abs(eval_i, E, bwidth):
    delta_fn = np.zeros_like(E)
    #print(Earray-eval_i, eval_i)
    i_where = np.argwhere(abs(E-eval_i)<bwidth)
    delta_fn[i_where] = 1
    return delta_fn

def gauss_abs(eval_i, E, bwidth):
    sol = np.exp(-(eval_i-E)**2/2/bwidth**2)
    return sol


