#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
RESERVOIR SIMULATION WITH MULTIGRID AND DEFLATED ITERATIVE SOLVERS
===============================================================================

Numerical implementation of finite volume methods with advanced iterative 
solvers for reservoir simulation in heterogeneous porous media.

Authors:
    Luis Antonio López Peña        <luisantondroid@gmail.com>
    Gabriela Berenice Díaz Cortés  <gbdiaz@gmail.com>


Created: 2024
Last Modified: August 2025

Description:
    This module implements advanced iterative solvers including:
    - Conjugate Gradient (CG) methods
    - Deflated Conjugate Gradient (DCG) 
    - Multigrid (MG) methods
    - Multigrid-preconditioned Conjugate Gradient (MGCG)
    
    Applied to reservoir simulation problems with:
    - Heterogeneous permeability fields
    - Distributed fracture networks
    - 2D and 3D computational domains

References:
    This work extends the deflation methods presented in:
    
    Díaz Cortés, G.B., Vuik, C., Jansen, J.D. (2021). Accelerating the 
    solution of linear systems appearing in two-phase reservoir simulation 
    by the use of POD-based deflation methods. Computational Geosciences, 
    25(5), 1621-1645. https://doi.org/10.1007/s10596-021-10070-z

Citation:
    If you use this code in your research, please cite the above reference
    and acknowledge this implementation.

===============================================================================
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csc_matrix, lil_matrix
from scipy import sparse
import numpy as np
import numpy.linalg as LA 
import pandas as pd

from Converters import *
from def_func import * 

# =============================================================================
# ------------------------- DataFrame Creation Functions ---------------------
# =============================================================================

def df_all(df, Lx, Ly, lay, perm_coef, case, x_true, *args):   
    """
    Create comprehensive DataFrame with all solver results
    
    Parameters:
    df: list to append DataFrames
    Lx, Ly: domain dimensions
    lay: number of layers
    perm_coef: permeability coefficients [min, max]
    case: case name
    x_true: exact solution
    *args: solver result objects
    """
    for arg in args:        
        for a in np.arange(0, len(arg.Nx)):
            m = {
                'N': [arg.Nx[a] * arg.Nx[a]], 
                'Nx': [arg.Nx[a]], 
                'Case': case, 
                'Method': arg.name,
                'layers': lay, 
                'Perm_min': perm_coef[0],
                'Perm_max': perm_coef[1], 
                'Perm_contrast': perm_coef[1] / perm_coef[0], 
                'Iterations': arg.Iter[a],
                'Residual': arg.rres[a], 
                'Error': arg.rerr[a],
                'Time_Initial_Process': arg.t_init[a], 
                'Time_Iterations_Process': arg.t_iter[a],
                'Total_Time_Process': arg.t_init[a] + arg.t_iter[a], 
                'Total_Time_Unit_Process': arg.t_unit[a],
                'Time_Initial': arg.tp_init[a], 
                'Time_Iterations': arg.tp_iter[a],
                'Total_Time': arg.tp_init[a] + arg.tp_iter[a], 
                'Total_Time_Unit': arg.t_unit[a],
                'Initial_Work': arg.w_init[a], 
                'Work_Iterations': arg.w_iter[a],
                'Total_Work_Iterations': arg.wt_iter[a],
                'Total_Work': arg.w_iter[a] + arg.w_init[a], 
                'Condition_Number': arg.cn[a]
            }
            m_1 = pd.DataFrame(m, index=None)
            df.append(m_1)
    return df

# =============================================================================
# ------------------------- Results Storage Functions -----------------------
# =============================================================================

def save_results(n_i, dic_results, *args):  
    """
    Save solver results to dictionary structure
    
    Parameters:
    n_i: iteration index
    dic_results: results dictionary
    *args: solver result objects
    """
    m = 0
    for arg in args:
        dic_results['Nx'][m, n_i] = arg.Nx
        dic_results['Iter'][m, n_i] = max(arg.Its)
        dic_results['T_unit'][m, n_i] = arg.t_unit * max(arg.Its) + max(arg.Its)
        dic_results['T_init'][m, n_i] = arg.t_init
        dic_results['Tp_init'][m, n_i] = arg.tp_init
        dic_results['W_init'][m, n_i] = arg.flops_init
        dic_results['T_iter'][m, n_i] = arg.t_iter * max(arg.Its)
        dic_results['Tp_iter'][m, n_i] = arg.tp_iter * max(arg.Its)
        dic_results['Rres'][m, n_i] = arg.rres[-1]
        dic_results['Rerr'][m, n_i] = arg.rerr[-1]
        dic_results['Mat_cn'][m, n_i] = arg.cn
        
        # Handle different work calculations for MG vs other methods
        if arg.name != 'MG ':
            dic_results['W_T_iter'][m, n_i] = arg.flops_iter * max(arg.Its)
            dic_results['W_iter'][m, n_i] = arg.flops_iter
        else:
            dic_results['W_iter'][m, n_i] = arg.flops_iter
            dic_results['W_T_iter'][m, n_i] = arg.flops_iter * max(arg.Its)
        m += 1
    return dic_results

def upd_results(dic_results, *args):  
    """
    Update solver objects with results from dictionary
    
    Parameters:
    dic_results: results dictionary
    *args: solver result objects to update
    """
    m = 0
    for arg in args:  
        arg.Nx = dic_results['Nx'][m, :]
        arg.Iter = dic_results['Iter'][m, :]
        arg.t_init = dic_results['T_init'][m, :]
        arg.t_unit = dic_results['T_unit'][m, :]
        arg.tp_init = dic_results['T_init'][m, :]
        arg.tp_unit = dic_results['T_unit'][m, :]
        arg.t_iter = dic_results['T_iter'][m, :]
        arg.tp_iter = dic_results['Tp_iter'][m, :]
        arg.w_init = dic_results['W_init'][m, :]
        arg.wt_iter = dic_results['W_T_iter'][m, :]
        arg.w_iter = dic_results['W_iter'][m, :]
        arg.rres = dic_results['Rres'][m, :]
        arg.rerr = dic_results['Rerr'][m, :]
        arg.cn = dic_results['Mat_cn'][m, :]
        m += 1

# =============================================================================
# ------------------------- Results Export Functions ------------------------
# =============================================================================

def results_df(dir_plot, case, *args):
    """
    Create and export timing results DataFrames
    
    Parameters:
    dir_plot: directory for output files
    case: case name
    *args: solver result objects
    
    Note: This function references undefined functions (results_tp, results_tt)
    which would need to be implemented separately
    """
    methods = []
    for arg in args:  
        methods.append(arg)        
        
        # Note: results_tp and results_tt functions need to be defined
        # These functions should create timing analysis DataFrames
        try:
            frames = [results_tp(case, f) for f in methods]
            result_p = pd.concat(frames)
            
            framest = [results_tt(case, f) for f in methods]
            result_t = pd.concat(framest)
            
            # Export results to CSV files
            result_t.to_csv(dir_plot + 'times_t.txt', header=True, index=True, sep='&')
            result_p.to_csv(dir_plot + 'times_p.txt', header=True, index=True, sep='&')
            
            return result_p, result_t
            
        except NameError as e:
            print(f"Warning: {e}")
            print("Functions 'results_tp' and 'results_tt' need to be implemented")
            return None, None

