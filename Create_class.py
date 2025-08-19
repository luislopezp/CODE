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
# =============================================================================
# ------------------------- Required Libraries -------------------------------
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import linalg as SLA
from scipy.sparse import csc_matrix, lil_matrix, spdiags
import numpy.linalg as LA 
import scipy

from Converters import *
from Create_matrix import *
from def_func import *


# =============================================================================
# ----------------------------- Grid Classes ---------------------------------
# =============================================================================

class Grid_New3D:
    """
    3D Grid class for reservoir simulation
    
    Stores all variables related to reservoir dimensions and converts to SI units
    """
    def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz, N, Cx, Cy, Cz, L_units):
        
        # Calculate cell face and center coordinates
        [xf_vec, yf_vec, zf_vec] = cell_f3D(Cx, Cy, Cz, Nx, Ny, Nz)
        [xc_vec, yc_vec, zc_vec] = cell_c3D(xf_vec, yf_vec, zf_vec, Nx, Ny, Nz)
        [xc_m, yc_m, zc_m] = xyz_mesh(xc_vec, yc_vec, zc_vec, Nx, Ny, Nz, N)
        
        # Calculate cell spacing
        dxc = dxcenter(xc_vec, Nx, Ny, N)
        dyc = dycenter(yc_vec, Nx, Ny, N)
        dzc = dzcenter(zc_vec, Nx, Ny, N)        
        dxf = dxface(xf_vec, Nx, Ny, Nz, N)
        dyf = dyface(yf_vec, Nx, Ny, Nz, N)
        dzf = dzface(zf_vec, Nx, Ny, Nz, N)
        
        # Calculate cell face areas
        Ax = Area_x(dyf, dzf, N)
        Ay = Area_y(dxf, dzf, N)
        Az = Area_z(dxf, dyf, N)
        
        # Grid dimensions
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.N = Nx * Ny * Nz
        self.Nl = self.Nx * self.Ny 
        
        # Domain dimensions (converted to SI)
        self.Lx = Length_converter(Lx, L_units)   
        self.Ly = Length_converter(Ly, L_units)
        self.Lz = Length_converter(Lz, L_units)
        
        # Mesh coordinates
        self.xfmesh = Cx
        self.yfmesh = Cy
        self.zfmesh = Cz
        self.xcmesh = xc_m
        self.ycmesh = yc_m
        self.zcmesh = zc_m
        
        # Cell coordinates
        self.xf = xf_vec
        self.yf = yf_vec
        self.zf = zf_vec
        self.xc = xc_vec
        self.yc = yc_vec
        self.zc = zc_vec
        
        # Cell spacing
        self.dxc = dxc
        self.dyc = dyc
        self.dzc = dzc
        self.dxf = dxf
        self.dyf = dyf
        self.dzf = dzf
        
        # Cell face areas
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        
        # Units
        self.L_units = 'm'

class Grid_NewMG:
    """
    Multigrid Grid class for reservoir simulation
    """
    def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz, Cx, Cy, L_units):
        
        [xf_vec, yf_vec, yf1_vec] = cell_f(Cx, Cy, Nx, Ny)
        [xc_vec, yc_vec] = cell_c(Cx, Cy, Nx, Ny)
        [dxf, dxc] = Gdx(xf_vec, xc_vec, Nx, Ny)
        [dyf, dyc] = Gdy(yf1_vec, yc_vec, Nx, Ny)
        
        # Domain dimensions (converted to SI)
        self.Lx = Length_converter(Lx, L_units)   
        self.Ly = Length_converter(Ly, L_units)
        self.Lz = Length_converter(Lz, L_units)
        
        # Mesh coordinates
        self.xfmesh = Cx
        self.yfmesh = Cy
        self.xf = xf_vec
        self.yf = yf_vec
        self.xc = xc_vec
        self.yc = yc_vec
        self.dxf = dxf
        self.dyf = dyf
        self.dxc = dxc
        self.dyc = dyc
        
        # Grid dimensions
        self.Nx = Nx + 1
        self.Ny = Ny + 1
        self.Nz = Nz
        self.Dx = Lx / Nx
        self.Dy = Ly / Ny
        self.Dz = Lz / Nz
        self.N = Nx * Ny * Nz
        self.Vol = self.Dx * self.Dy * self.Dz 
        self.L_units = 'm'

# =============================================================================
# ------------------------------ Rock Class ----------------------------------
# =============================================================================

class Rock:
    """
    Rock properties class
    
    Stores all variables related to rock properties and converts to SI units
    """
    def __init__(self, K, K_units, poro):
        self.perm = Permeability_converter(K, K_units)
        self.perm_units = 'm2'
        self.poro = poro

def Class_Rock(rock_c):
    """
    Create Rock class from input dictionary
    
    Parameters:
    rock_c: dictionary containing rock properties
    """
    poro = rock_c['poro'][0]      # Porosity    
    K = rock_c['K'][0]            # Permeability
    K_units = rock_c['K'][1]      # Permeability units

    # Create rock class with given parameters
    rock = Rock(K, K_units, poro)
    return rock

# =============================================================================
# ------------------------------ Fluid Class ---------------------------------
# =============================================================================

class Fluid:
    """
    Fluid properties class
    
    Stores all variables related to fluid properties and converts to SI units
    """
    def __init__(self, rho, rho_units, mu, mu_units):
        self.rho = Density_converter(rho, rho_units)
        self.rho_units = 'kg/m3'
        self.mu = Viscosity_converter(mu, mu_units)
        self.mu_units = 'Pa'

def Class_Fluid(Fluid_c):
    """
    Create Fluid class from input dictionary
    
    Parameters:
    Fluid_c: dictionary containing fluid properties
    """
    rho = Fluid_c['rho'][0]        # Density
    rho_units = Fluid_c['rho'][1]  # Density units    
    mu = Fluid_c['mu'][0]          # Viscosity
    mu_units = Fluid_c['mu'][1]    # Viscosity units

    # Create fluid class with given parameters
    fluid = Fluid(rho, rho_units, mu, mu_units)  
    return fluid 

# =============================================================================
# ------------------------ Boundary Conditions Classes ----------------------
# =============================================================================

class BC:
    """
    2D Boundary Conditions class
    
    Stores all variables related to boundary conditions and converts to SI units
    """
    def __init__(self, BL, BR, BN, BS, BL_units, BR_units, BN_units, 
                 BS_units, BL_type, BR_type, BN_type, BS_type):              
        # Left boundary
        if BL_type == 'D':
            self.BL = Pressure_converter(BL, BL_units)
            self.BL_units = 'Pa'            
        else: 
            self.BL = Flux_converter(BL, BL_units) 
            self.BL_units = 'm3s'            
        self.BL_type = BL_type
        
        # Right boundary
        if BR_type == 'D':
            self.BR = Pressure_converter(BR, BR_units)
            self.BR_units = 'Pa'  
        else: 
            self.BR = Flux_converter(BR, BR_units) 
            self.BR_units = 'm3s' 
        self.BR_type = BR_type
        
        # North boundary
        if BN_type == 'D':
            self.BN = Pressure_converter(BN, BN_units)
            self.BN_units = 'Pa'
        else: 
            self.BN = Flux_converter(BN, BN_units) 
            self.BN_units = 'm3s'
        self.BN_type = BN_type 
        
        # South boundary
        if BS_type == 'D':
            self.BS = Pressure_converter(BS, BS_units)
            self.BS_units = 'Pa'
        else: 
            self.BS = Flux_converter(BS, BS_units)
            self.BS_units = 'm3s'
        self.BS_type = BS_type

class BC_3D:
    """
    3D Boundary Conditions class
    
    Stores all variables related to 3D boundary conditions and converts to SI units
    """
    def __init__(self, BL, BR, BN, BS, BU, BD, BL_units, BR_units, BN_units, 
                 BS_units, BU_units, BD_units, BL_type, BR_type, BN_type, 
                 BS_type, BU_type, BD_type):              
        # Left boundary
        if BL_type == 'D':
            self.BL = Pressure_converter(BL, BL_units)
            self.BL_units = 'Pa'            
        else: 
            self.BL = Flux_converter(BL, BL_units) 
            self.BL_units = 'm3s'            
        self.BL_type = BL_type
        
        # Right boundary
        if BR_type == 'D':
            self.BR = Pressure_converter(BR, BR_units)
            self.BR_units = 'Pa'  
        else: 
            self.BR = Flux_converter(BR, BR_units) 
            self.BR_units = 'm3s' 
        self.BR_type = BR_type
        
        # North boundary
        if BN_type == 'D':
            self.BN = Pressure_converter(BN, BN_units)
            self.BN_units = 'Pa'
        else: 
            self.BN = Flux_converter(BN, BN_units) 
            self.BN_units = 'm3s'
        self.BN_type = BN_type 
        
        # South boundary
        if BS_type == 'D':
            self.BS = Pressure_converter(BS, BS_units)
            self.BS_units = 'Pa'
        else: 
            self.BS = Flux_converter(BS, BS_units)
            self.BS_units = 'm3s'
        self.BS_type = BS_type
        
        # Upper boundary
        if BU_type == 'D':
            self.BU = Pressure_converter(BU, BU_units)
            self.BU_units = 'Pa'
        else: 
            self.BU = Flux_converter(BU, BU_units)
            self.BU_units = 'm3s'
        self.BU_type = BU_type
        
        # Bottom boundary
        if BD_type == 'D':
            self.BD = Pressure_converter(BD, BD_units)
            self.BD_units = 'Pa'
        else: 
            self.BD = Flux_converter(BD, BD_units)
            self.BD_units = 'm3s'
        self.BD_type = BD_type

# =============================================================================
# ------------------------ Eigenvalue Analysis Classes ----------------------
# =============================================================================

class PMA_c:
    """Preconditioned Matrix with Deflation eigenvalue analysis"""
    def __init__(self, A, M, Z, G):
        self.n = G.N
        self.Nx = G.Nx
        self.name = 'PMA'
        self.marker = '*'
        self.mat = csc_matrix(PA_f(MA_f(A, M), Z))
        self.eig = np.sort(np.real(SLA.eigs(self.mat, self.n-2, return_eigenvectors=False))) 
        self.leig = self.eig[np.shape(Z)[1]:]
        self.lmax = max(self.leig)
        self.lmin = min(self.leig)
        self.cn = self.lmax / self.lmin

class PA_c:
    """Matrix with Deflation eigenvalue analysis"""
    def __init__(self, A, Z, G):
        self.n = G.N
        self.Nx = G.Nx
        self.name = 'PA'
        self.marker = 'x'
        self.mat = csc_matrix(PA_f(A, Z))
        self.eig = np.sort(np.real(SLA.eigs(self.mat, self.n-2, return_eigenvectors=False)))
        self.leig = self.eig[np.shape(Z)[1]:]
        self.lmax = max(self.leig)
        self.lmin = min(self.leig)
        self.cn = self.lmax / self.lmin

class A_c:
    """Original Matrix eigenvalue analysis"""
    def __init__(self, A, G):
        self.n = G.N
        self.Nx = G.Nx
        self.name = 'A'
        self.marker = 'd'
        self.mat = csc_matrix(A)
        self.eig = np.sort(np.real(SLA.eigs(self.mat, self.n-2, return_eigenvectors=False)))
        self.lmax = max(self.eig)
        self.lmin = min(self.eig)
        self.cn = self.lmax / self.lmin

class MA_c:
    """Preconditioned Matrix eigenvalue analysis"""
    def __init__(self, A, M, G):
        self.n = G.N
        self.Nx = G.Nx
        self.name = 'MA'
        self.marker = 'o'
        self.mat = csc_matrix(MA_f(A, M))
        self.eig = np.sort(np.real(SLA.eigs(self.mat, self.n-2, return_eigenvectors=False)))
        self.lmax = max(self.eig)
        self.lmin = min(self.eig)
        self.cn = self.lmax / self.lmin

# =============================================================================
# ------------------------- Eigenvalue Utility Functions --------------------
# =============================================================================

def MA_f(A, M):
    """Create preconditioned matrix M*A"""
    MA = M * A
    MA = csc_matrix(MA)
    return MA

def PA_f(A, Z):
    """Create deflated matrix"""
    V = A * Z
    E = Z.transpose() * V
    EI = SLA.inv(E)
    B = V * EI
    PA = Deflation_P(B, Z, A)
    PA = csc_matrix(PA)
    return PA    

def get_eigs(A_S, G, dv):
    """
    Calculate eigenvalues for different matrix configurations
    
    Parameters:
    A_S: system matrix
    G: grid object
    dv: deflation vector type
    """
    if G.Nx < 1000:
        Z = G.Z 
        d = 1 / A_S.diagonal()
        M = spdiags(d, 0, G.N, G.N)
        Mat_A = A_c(A_S, G)
        Mat_PA = PA_c(A_S, Z, G)
        Mat_MA = MA_c(A_S, M, G)
        Mat_PMA = PMA_c(A_S, M, Z, G)
    return Mat_A, Mat_MA, Mat_PA, Mat_PMA     

def results_cn(cas, dir_plot, *args):  
    """
    Save condition number results to file
    
    Parameters:
    cas: case name
    dir_plot: directory for plots
    *args: eigenvalue analysis objects
    """
    for arg in args:  
        a1 = []
        a2 = []
        a3 = []
        a4 = []
        a5 = []
        for arg in args:
            a1.append(arg.name)
            a2.append(arg.lmax)
            a3.append(arg.lmin)
            a4.append(arg.cn)
            a5.append(arg.Nx)
        n = arg.n
        
        m = {'Case': cas, 'Nx': a5, 'Mat': a1, 'lmax': a2, 'lmin': a3, 'eff cn': a4}
        m_pd = pd.DataFrame(m)
        m_pd.to_csv(dir_plot + 'cn.txt', header=True, index=True, sep='&')
        m_pd.to_pickle(dir_plot + 'cn' + cas + '.pkl')
    return m_pd

def df_cn_all(plot_dir, df, Lx, Ly, perm_coef, cas, *args):   
    """
    Create comprehensive condition number DataFrame
    
    Parameters:
    plot_dir: directory for plots
    df: list to append DataFrames
    Lx, Ly: domain dimensions
    perm_coef: permeability coefficients
    cas: case name
    *args: eigenvalue analysis objects
    """
    for arg in args:        
        m = {'Lx': Lx, 'Ly': Ly, 'N': [arg.Nx * arg.Nx], 'Nx': [arg.Nx],
             'Perm_min': perm_coef[0], 'Perm_max': perm_coef[1], 
             'Perm_cont': perm_coef[1] / perm_coef[0],
             'Case': cas, 'Mat': arg.name, 'lmax': arg.lmax,
             'lmin': arg.lmin, 'eff cn': arg.cn}
        m_1 = pd.DataFrame(m, index=None)
        df.append(m_1)
    return df