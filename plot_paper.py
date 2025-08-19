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
import os
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
# --------------------------- Plot Configuration -----------------------------
# =============================================================================

# Global plot settings
del_figs = False  # Set to True to automatically close figures after saving

# Color and line style definitions
colors = ["red", "blue", "green", "darkviolet", "goldenrod", "maroon", 
          "royalblue", "navy", "magenta", "purple", "orchid", "pink", "orange"]
lines = [':', '-.', '--', '-']

# =============================================================================
# --------------------------- 3D Plotting Functions -------------------------
# =============================================================================

def plot_3D(p, G, x, y,  z_label=False):
    """
    Create 3D surface plot with appropriate limits and viewing angle

    Parameters:
    p: array of calculated pressure field
    G: grid object
    x: array of x coordinates
    y: array of y coordinates  
    z_label: whether to show z-axis label
    """
    cm = 1/2.54  # centimeters to inches conversion
    fig = plt.figure(figsize=(8*cm, 5*cm))
    p = p.reshape(G.Nx, G.Nx)  # Convert array to matrix
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, p, rstride=1, cstride=1, cmap='viridis', 
                   edgecolor='none', alpha=1)
    ax.set_xlabel('$X$', fontsize=8)
    ax.set_ylabel('$Y$', fontsize=8)
    
    if z_label:
        ax.set_zlabel(r'$P$', fontsize=8)
    else:
        ax.set_zlabel('  ', fontsize=8)
        
    ax.xaxis.set_tick_params(labelsize=8, rotation=90)
    ax.yaxis.set_tick_params(labelsize=8, rotation=90)
    ax.zaxis.set_tick_params(labelsize=8, rotation=30)
    ax.view_init(30, 220)
    plt.tight_layout()
    
    if del_figs:
        plt.clf()
        plt.close(fig)


# =============================================================================
# --------------------------- 2D Solution Comparison ------------------------
# =============================================================================

def Compare_sols2D(case, G, df_sol, layers,  x_true):
    """
    Compare 2D solutions for different methods
    
    Parameters:
    case: case name
    G: grid object
    df_sol: DataFrame with solutions from different methods
    layers: number of layers
    x_true: exact solution
    """
    cm = 1/2.54  # centimeters to inches
    x = G.xcmesh
    y = G.ycmesh 
    X, Y = np.meshgrid(x, y)

    methods = df_sol.columns
            
    for method in methods: 
        sol = np.array(df_sol[method])
        dic = {'Case': case, 'Method': method, 'Nx': G.Nx, 
               'x_it': sol, 'x_true': x_true}
        name = dic['Case'] + '_' + dic['Method'] 
        
        # True solution plot
        plot_3D(Pa2psi(dic['x_true']), G, x, y, True)
                
        # Iterative solution plot
        plot_3D(Pa2psi(dic['x_it']), G, x, y, True)
                
        # Difference plot
        plot_3D(Pa2psi(dic['x_true'] - dic['x_it']), G, x, y, True)

        # Y-direction pressure profile
        fig = plt.figure(figsize=(8*cm, 5*cm))
        plt.plot(y, Pa2psi(sol[0:G.N:G.Nx]), '*')
        plt.xlabel('Y [m]', fontsize=8) 
        plt.ylabel('Pressure [psi]', fontsize=8)
        plt.yticks(fontsize=7)
        plt.xticks(fontsize=7)

        if del_figs:
            plt.close(fig)
        
        # 2D contour plot
        fig = plt.figure(figsize=(6*cm, 6*cm))
        ax = fig.add_subplot(111)
        sol = Pa2psi(sol)
        P = sol.reshape(G.Ny, G.Nx)
        plt.pcolormesh(X, Y, P) 
        plt.xlabel('X', fontsize=8) 
        plt.ylabel('Y', fontsize=8)
        plt.yticks(fontsize=7)
        plt.xticks(fontsize=7)
        ax.set_aspect('equal', adjustable='box')
        if del_figs:
            plt.close(fig)

# =============================================================================
# --------------------------- Convergence Plots -----------------------------
# =============================================================================

def Compare_rres(G, df_rres, layers, markers):  
    """
    Compare relative residuals for different methods
    
    Parameters:
    G: grid object
    df_rres: DataFrame with residual data
    layers: number of layers
    markers: dictionary of markers for each method
    """
    cm = 1/2.54  # centimeters to inches
    fig = plt.figure(figsize=(8*cm, 5*cm))
    methods = df_rres.columns
    i = 0
    
    for method in methods: 
        rres = df_rres[method][~df_rres[method].isna()]
        its = np.arange(len(rres))
        plt.plot(its, rres, marker=markers[method], ls=lines[0], 
                color=colors[i], label=method, markersize=3)
        i += 1

    plt.xlabel('Iteration', fontsize=8) 
    plt.ylabel('Relative residual', fontsize=8)
    plt.yscale('log')
    plt.yticks(fontsize=7)
    plt.xticks(fontsize=7)
    plt.ylim([1e-9, 1e4])
    plt.legend(loc='upper right', fontsize=5)
    
    if del_figs:
        plt.close(fig)
        
def Compare_rerr(G, df_rerr, layers,  markers):  
    """
    Compare relative errors for different methods
    
    Parameters:
    G: grid object
    df_rerr: DataFrame with error data
    layers: number of layers
    markers: dictionary of markers for each method
    """
      
    cm = 1/2.54  # centimeters to inches
    fig = plt.figure(figsize=(8*cm, 5*cm))
    methods = df_rerr.columns
    i = 0
    
    for method in methods: 
        rerr = df_rerr[method][~df_rerr[method].isna()]
        its = np.arange(len(rerr))
        plt.plot(its, rerr, marker=markers[method], ls=lines[0], 
                color=colors[i], label=method, markersize=3)
        i += 1
        
    plt.xlabel('Iteration', fontsize=8) 
    plt.ylabel('Relative error', fontsize=8)
    plt.yscale('log')
    plt.yticks(fontsize=7)
    plt.xticks(fontsize=7)
    plt.ylim([1e-9, 5])
    plt.legend(loc='upper right', fontsize=5)
    if del_figs:
        plt.close(fig)