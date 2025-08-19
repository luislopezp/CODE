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
from scipy.sparse import *
from scipy import *
import numpy as np
from scipy.sparse import linalg as SLA

def def_vect(dv, G):
    """
    Generate deflation vectors based on specified type
    
    Parameters:
    dv: deflation vector type ('SD' for subdomain, 'Eigs' for eigenvectors)
    G: grid object
    
    Returns:
    Z: deflation matrix
    """
    if dv == 'SD':
        Z = lil_matrix((G.Nx * G.Ny, 4))
        nz = int(G.N / 4)
        for i in [0, 1, 2, 3]:
            Z[i*nz:(i+1)*nz, i] = 1
            
    elif dv == 'Eigs':   
        Mat_MA = MA_f(A, M, M2)
        Mat_MA = csc_matrix(Mat_MA)
        eival, Z = SLA.eigsh(Mat_MA, G.N-1, return_eigenvectors=True)  
        Z = lil_matrix(Z[:, 900:])
        
    return Z

def Z_sub(Nx, Z_vec):
    """
    Compute subdomain deflation matrix for square grid
    
    Parameters:
    Nx: grid size in one direction
    Z_vec: number of deflation vectors
    
    Returns:
    Z: sparse deflation matrix
    """
    N = Nx * Nx
    Z_el = int(N / Z_vec)
    
    row = [x for x in range(0, N)]
    col = []
    for i in range(0, Z_vec):
        col.append(i * np.ones(Z_el))
    col = [y for x in col for y in x]
    data = np.ones(N)
    Z = csr_matrix((data, (row, col)), shape=(N, Z_vec))
    
    return Z

def Deflation_P(B, Z, r, T=False):
    """
    Apply deflation projection operator P
    
    Parameters:
    B: deflation matrix B = V * E^(-1)
    Z: deflation vectors
    r: vector to deflate
    T: transpose flag
    
    Returns:
    s: deflated vector
    """
    if T == True:
        s = r - Z.dot(B.transpose().dot(r))
    else:
        s = r - B.dot(Z.transpose().dot(r))
    
    return s

def Correction_Q(Z, EI, x):
    """
    Apply deflation correction operator Q
    
    Parameters:
    Z: deflation vectors
    EI: inverse of E matrix (E^(-1))
    x: vector to correct
    
    Returns:
    q: correction vector
    """
    q = Z.dot(EI.dot(Z.transpose().dot(x)))
    
    return q