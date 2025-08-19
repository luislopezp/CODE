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
import sys
import json
from datetime import datetime
import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import linalg as SLA
from scipy.linalg import eigh

# Custom libraries 
from CGM import *
from MGM import *
from Create_class import *
from Create_matrix import *
from results_analysis import *
from plot_paper import *

# Time configuration
cdt = datetime.now()
n_time = str(cdt.year) + "_" + str(cdt.month) + '_' + str(cdt.day)

# Configuration
True_sol = True
Comp_Eigs = False  
plot_res = True  
fracture = False

# Only D_frac case
cases_name = ['D_frac']
Nfrac = 8  # Fixed at 8 fractures

# Initialization
init = ["random","DCG","DPCG"]
x_init = init[0]

#%% 
# =============================================================================
# -------------------------------- Reservoir ----------------------------------
# =============================================================================
# --------------------------------- Rock -------------------------------------
# =============================================================================
K_units = 'Da'
poro = 0.25 
# =============================================================================
# -------------------------------- Fluid ------------------------------------
# =============================================================================
rho = 1
rho_units = 'lbft3'
mu = 0.51 
mu_units = 'cp'
#%% 
# =============================================================================
# ------------------------------- Dimensions --------------------------------
# =============================================================================

# 32x32 grid
Nx = 32
Ny = 32
Nz = 1

Lx = 762
Ly = 762  
Lz = 762
L_units = 'm'



# =============================================================================
# -------------------------- Boundary conditions ----------------------------
# =============================================================================
BL_type  = 'N'
BL_value = 0
BL_units = 'stbday'

BR_type  = 'N'
BR_value = 0
BR_units ='stbday' 

BN_type  = 'D'
BN_value = 8000 
BN_units = 'psi'

BS_type  = 'D'
BS_value = 0
BS_units = 'psi'

BU_type  = 'N'
BU_value = 0
BU_units = 'stbday'

BD_type  = 'N'
BD_value = 0
BD_units ='stbday'

#%% 
# =============================================================================
# -------------------------------- Solvers -----------------------------------
# =============================================================================

# All specified methods
methods = ["DCG","MG","MGCG","MGCG1","DPCG"]

dic_methods = {}
N_methods = len(methods)+1

# Stopping criteria
tol = 5e-7
MaxIter = 800

# MG smoother iterations
S1_it = 10
S2_it = 10

# Deflation vectors
def_vec = ['SD','Eigs']
dv = def_vec[0]

# Permeabilities for D_frac
perm = [500, 0.001]  # [high, low] for distributed fractures
#%% 
# =============================================================================
# -------------------------------- Simulation ---------------------------------
# =============================================================================

print("Case: D_frac with", Nfrac, "fractures")
print("Grid:", Nx, "x", Ny)

# Case configuration
c_case = "D_frac"
perm_coef = [perm[0], perm[1]]
layers = int(Nfrac*2)

# Grid construction
Cx = np.linspace(0, Lx, Nx+1)
Cy = yfracture(Ny, Nfrac, Ly)  # Special grid for fractures
Cz = np.linspace(0,Lz,Nz+1)

N = Nx*Ny*Nz

# Create Grid object
G = Grid_New3D(Lx, Ly, Lz, Nx, Ny, Nz, N, Cx, Cy, Cz, L_units)

# Boundary conditions
bc = BC_3D(BL_value*np.ones(G.N), BR_value*np.ones(G.N), 
         BN_value*np.ones(G.N), BS_value*np.ones(G.N),
         BU_value*np.ones(G.N), BD_value*np.ones(G.N),
         BL_units, BR_units, BN_units, BS_units, BU_units, BD_units, 
         BL_type, BR_type, BN_type, BS_type, BU_type, BD_type)   

# Fluid
Fluid_c = {'rho':[rho* np.ones(G.N) ,rho_units],'mu':[mu* np.ones(G.N) ,mu_units]}
fluid = Class_Fluid(Fluid_c)

# Permeability and rock
K, G.Z = Perm(G, c_case, perm_coef, Nfrac, layers)
rock_c = {'poro':[poro* np.ones(G.N)],'K':[K,K_units]}
rock = Class_Rock(rock_c)

# System matrix construction
print("Building system matrix...")
[A, b, Tx, Ty, Tz] = AI_mat_full_FV_3D(G,rock,fluid,bc)

A_S = csc_matrix(A)  

# Eigenvalue calculation if required
if Comp_Eigs:
    Mat_A,Mat_MA,Mat_PA,Mat_PMA = get_eigs(A_S,G,dv)

# Exact solution
print("Computing exact solution...")

x_true =  SLA.spsolve(A_S, b)

# Initialization
print("Preparing initialization:", x_init)
if x_init == "random":
    x_0 = np.random.rand(np.size(b))   
elif x_init == "DCG":
    x_0 = np.random.rand(np.size(b)) 
    DCG_GD = DGC(A_S,b,x_0,x_true,G,1,tol,dv)
    x_0 = DCG_GD.x
elif x_init == "DPCG":
    x_0 = np.random.rand(np.size(b)) 
    DPCG_GD = DPGC(A_S,b,x_0,x_true,G,1,'J',tol,dv)
    x_0 = DPCG_GD.x

# Execute methods
print("Running numerical methods...")
for method in methods:
    print(f"  - Executing {method}...")
    
    if method == "DCG":
        DCG_GD = DGC(A_S,b,x_0,x_true,G,MaxIter,tol,dv)
        dic_methods[method] = DCG_GD 
        if Comp_Eigs:
            DCG_GD.cn = Mat_PA.cn
        else:
            DCG_GD.cn = 0
            
    elif method == "DPCG":
        DPCG_GD = DPGC(A_S,b,x_0,x_true,G,MaxIter,'J',tol,dv)
        dic_methods[method] = DPCG_GD 
        if Comp_Eigs:
            DPCG_GD.cn = Mat_PMA.cn
        else:
            DPCG_GD.cn = 0
            
    elif method == "MG":
        if Comp_Eigs:
            MG_GD = MG(A_S,b,x_0,x_true,G,S1_it, S2_it, MaxIter,tol,Comp_Eigs)
        else:
            MG_GD = MG(A_S,b,x_0,x_true,G,S1_it, S2_it, MaxIter,tol)
        dic_methods[method] = MG_GD 
        if Comp_Eigs:
            MG_GD.cn = np.linalg.cond(np.load("A_2h.npy",allow_pickle=True))
        else:
            MG_GD.cn = 0
            
    elif method == "MGCG":
        MGCG_GD = MGCG(A_S,b,x_0,x_true,G,MaxIter,tol,S1_it,S2_it)
        dic_methods[method] = MGCG_GD 
        if Comp_Eigs:
            MGCG_GD.cn = 0
        else:
            MGCG_GD.cn = 0
            
    elif method == "MGCG1":
        MGCG1_GD = MGCG1(A_S,b,x_0,x_true,G,MaxIter,tol,S1_it,S2_it)
        dic_methods[method] = MGCG1_GD 
        if Comp_Eigs:
            MGCG1_GD.cn = 0
        else:
            MGCG1_GD.cn = 0

# Prepare data for visualization only (no permanent files)
print("Preparing data for visualization...")

# DataFrames for residuals, errors and solutions (memory only)
df_rres = []
df_rerr = []
df_sol = []
markers = {}

for method in methods:
    df_rres.append(dic_methods[method].rres)
    df_rerr.append(dic_methods[method].rerr)
    df_sol.append(dic_methods[method].x)
    markers[method] = dic_methods[method].marker

# Convert to DataFrames (only for plots)
df_rres = pd.DataFrame(df_rres).T
df_rres.columns = methods    
df_rerr = pd.DataFrame(df_rerr).T
df_rerr.columns = methods
df_sol = pd.DataFrame(df_sol).T
df_sol.columns = methods

# Generate plots (display only, no saving)
if plot_res:
    print("Generating plots...")
    
    # 2D solution plots
    Compare_sols2D(c_case, G, df_sol, layers,  x_true)
    
    # Convergence plots (residuals)
    Compare_rres(G, df_rres, layers,  markers)
    
    # Error plots
    Compare_rerr(G, df_rerr, layers, markers)
    
    print("Plots displayed")


print("Simulation completed successfully!")

# Show results summary
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"Case: {c_case} with {Nfrac} fractures")
print(f"Grid: {Nx} x {Ny}")
print(f"Permeabilities: {perm[0]} / {perm[1]} (contrast: {perm[0]/perm[1]:.0f})")
print(f"Tolerance: {tol}")
print(f"Maximum iterations: {MaxIter}")
print()

for method in methods:
    solver = dic_methods[method]
    final_iter = len(solver.rres) - 1
    print(f"{method:>8}: {final_iter:>3} its, "
          f"rres={solver.rres[-1]:.2e}, "
          f"rerr={solver.rerr[-1]:.2e}")