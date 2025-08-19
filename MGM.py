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

import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import linalg as SLA
from numpy import linalg as LA
import sys
from scipy.sparse import csc_matrix, lil_matrix, diags
from scipy import sparse
import os
import time
from flops_op import *

# =============================================================================
# --------------------------- Grid Classes -----------------------------------
# =============================================================================

class Grid_VC:
    """
    Grid class for V-Cycle operations (unified 2D/3D)
    
    Automatically detects 2D vs 3D based on n_z parameter
    """
    def __init__(self, n_x, n_y, n_z, L):
        self.N_x = int(n_x / 2)
        self.N_y = int(n_y / 2)
        self.n_x = n_x
        self.n_y = n_y
        self.L = L
        self.Nxs = []
        self.Ls = []
        
        # Handle 2D vs 3D cases
        if n_z > 1:  # 3D case
            self.n_z = n_z
            self.N_z = int(n_z / 2)
            self.n = self.n_x * self.n_y * self.n_z
            self.N = self.N_x * self.N_y * self.N_z
            self.is_3d = True
        else:  # 2D case
            self.n_z = 1
            self.N_z = 1
            self.n = self.n_x * self.n_y
            self.N = self.N_x * self.N_y
            self.is_3d = False

# =============================================================================
# --------------------------- Result Classes ---------------------------------
# =============================================================================

class TGC:
    """Two Grid method results"""
    def __init__(self, x, S1_Its, S2_Its, Its, t, tp, rr, G, flops, x_true=[], re=[]):
        self.name = 'TG'
        self.marker = '>'
        self.x = x
        self.t_init = t[0]
        self.t_iter = t[1]
        self.tp_init = tp[0]
        self.tp_iter = tp[1]
        self.S1_Its = S1_Its
        self.S2_Its = S2_Its
        self.Its = Its
        self.rres = rr
        self.Nx = G.Nx    
        self.x_true = x_true       
        self.rerr = re     
        self.flops_init = float(flops[0])
        self.flops_iter = float(flops[1])

class MGC:
    """Multigrid method results"""
    def __init__(self, x, S1_Its, S2_Its, Its, t, tp, rr, r_true, G, flops, L, Nx_L, x_true, re):
        self.name = 'MG'
        self.marker = '>'
        self.x = x
        self.t_unit = t[2]
        self.t_init = t[0]
        self.t_iter = t[1]
        self.tp_init = tp[0]
        self.tp_iter = tp[1]
        self.Its = Its
        self.rres = rr
        self.r_true = r_true 
        self.Nx = G.Nx    
        self.x_true = x_true       
        self.rerr = re
        self.L = L
        self.Nx_L = Nx_L
        self.flops_init = float(flops[0])
        self.flops_iter = float(flops[1])

class MGCGS:
    """Multigrid Conjugate Gradient method results"""
    def __init__(self, x, Its, t, tp, rr, r_true, G, flops, L, Nx_L, x_true=[], re=[]):
        self.name = 'MGCG'
        self.marker = 's'
        self.x = x
        self.t_unit = t[2]
        self.t_init = t[0]
        self.t_iter = t[1]
        self.tp_init = tp[0]
        self.tp_iter = tp[1]
        self.Its = Its
        self.rres = rr
        self.r_true = r_true 
        self.Nx = G.Nx    
        self.x_true = x_true       
        self.rerr = re
        self.L = L
        self.Nx_L = Nx_L
        self.flops_init = float(flops[0])
        self.flops_iter = float(flops[1])

class MGCGS1:
    """Multigrid Conjugate Gradient method results (variant 1)"""
    def __init__(self, x, Its, t, tp, rr, r_true, G, flops, L, Nx_L, x_true=[], re=[]):
        self.name = 'MGCG1'
        self.marker = 's'
        self.x = x
        self.t_unit = t[2]
        self.t_init = t[0]
        self.t_iter = t[1]
        self.tp_init = tp[0]
        self.tp_iter = tp[1]
        self.Its = Its
        self.rres = rr
        self.r_true = r_true 
        self.Nx = G.Nx    
        self.x_true = x_true       
        self.rerr = re
        self.L = L
        self.Nx_L = Nx_L
        self.flops_init = float(flops[0])
        self.flops_iter = float(flops[1])

# =============================================================================
# --------------------------- Utility Functions -----------------------------
# =============================================================================

def init(a, f, x_0, n_x, n_y, n_z, L):
    """
    Initialize matrices and grid for multigrid operations
    
    Parameters:
    a: system matrix
    f: right-hand side
    x_0: initial guess
    n_x, n_y, n_z: grid dimensions
    L: multigrid level
    """
    A = a.copy()
    b = f.copy()
    u = x_0.copy()
    G_VC = Grid_VC(n_x, n_y, n_z, L)
    return A, b, u, G_VC

def presmoothing(S1_it, A, b, u, G):
    """
    Apply pre-smoothing using Jacobi iterations
    
    Parameters:
    S1_it: number of smoothing iterations
    A: system matrix
    b: right-hand side
    u: current solution
    G: grid object
    """
    Mv = A.diagonal()
    M_Inv = 0.2 * diags(1/Mv)
    M_IS = csc_matrix(M_Inv)   
    I_h = sparse.eye(G.n)
    I_hS = csc_matrix(I_h)
    
    for i in range(0, S1_it):              
        S_h = I_hS - M_IS.dot(A)
        s_h = M_IS.dot(b.T)
        uv1 = S_h.dot(u) + s_h
        u = uv1.copy()
    return u

def PR(G):
    """
    Create prolongation and restriction operators (unified 2D/3D)
    
    Parameters:
    G: grid object
    
    Returns:
    P: prolongation operator
    R: restriction operator
    """
    # X-direction operators
    Ix = lil_matrix((G.N_x, G.n_x))
    k = 0
    s = np.array([1, 1]).reshape(1, 2)
    for i in range(0, G.N_x):
        Ix[i, k:k+2] = s
        k = k + 2
    
    # Y-direction operators
    Iy = lil_matrix((G.N_y, G.n_y))
    k = 0
    for i in range(0, G.N_y):
        Iy[i, k:k+2] = s
        k = k + 2
    
    # Build operators based on dimensionality
    if G.is_3d:  # 3D case
        # Z-direction operators
        Iz = lil_matrix((G.N_z, G.n_z))
        k = 0
        for i in range(0, G.N_z):
            Iz[i, k:k+2] = s
            k = k + 2
        
        # 3D Kronecker products
        Rt = sparse.kron(sparse.kron(1/2*Ix, 1/2*Iy), 1/2*Iz)
        Pr = 2 * Rt.T
    else:  # 2D case
        # 2D Kronecker product
        Rt = sparse.kron(1/2*Ix, 1/2*Iy)
        Pr = 2 * Rt.T
    
    P = csc_matrix(Pr)
    R = csc_matrix(Rt)
    return P, R

def restrict(r_h, A_h, P, R, G):
    """
    Restrict residual and operator to coarser grid
    
    Parameters:
    r_h: fine grid residual
    A_h: fine grid matrix
    P: prolongation operator
    R: restriction operator
    G: grid object
    """
    r_2h = np.zeros(G.N)
    A_2h = lil_matrix((G.N, G.N))
    A_h_2h = lil_matrix((G.n, G.N))
    r_2h = R.dot(r_h)    
    A_h_2h = A_h.dot(P)  
    A_2h = R.dot(A_h_2h) 
    return r_2h, A_2h

def prolongation(e2h, P, G):
    """
    Prolongate error from coarse to fine grid
    
    Parameters:
    e2h: coarse grid error
    P: prolongation operator
    G: grid object
    """
    eh = np.zeros(G.n)
    eh = P.dot(e2h)
    return eh

def correction(eh, u, G):
    """
    Apply coarse grid correction
    
    Parameters:
    eh: error from coarse grid
    u: current solution
    G: grid object
    """
    un = np.zeros(G.n)
    un = u + eh
    return un

def postsmoothing(S2_it, A, b, G, un):
    """
    Apply post-smoothing using Jacobi iterations
    
    Parameters:
    S2_it: number of smoothing iterations
    A: system matrix
    b: right-hand side
    G: grid object
    un: solution after correction
    """
    Mv = A.diagonal()
    M_Inv = 0.2 * diags(1/Mv)
    M_IS = csc_matrix(M_Inv)   
    I_h = sparse.eye(G.n)
    I_hS = csc_matrix(I_h)
    
    for i in range(0, S2_it):
        S_h = I_hS - M_IS.dot(A)
        s_h = M_IS.dot(b)
        uv2 = S_h.dot(un) + s_h   
    return uv2

def v_cicle(S1_it, S2_it, A, b, u, G, Nx_L):
    """
    Recursive V-cycle implementation (unified 2D/3D)
    
    Parameters:
    S1_it: pre-smoothing iterations
    S2_it: post-smoothing iterations
    A: system matrix
    b: right-hand side
    u: initial guess
    G: grid object
    Nx_L: list of grid sizes at each level
    """
    G.L = G.L + 1
    
    # Initialize for current level
    A, b, u, G_VC = init(A, b, u, G.n_x, G.n_y, G.n_z, G.L)   
    u = presmoothing(S1_it, A, b, u, G_VC)    
    Nx_L.append(G.N_x)
    
    # Compute residual
    r_h = b - A.dot(u)
    P, R = PR(G_VC)     
    
    # Restrict to coarse grid
    r_2h, A_2h = restrict(r_h, A, P, R, G_VC)
    
    # Solve on coarsest grid or recurse
    if G_VC.N_x < 20:
        e2h = SLA.spsolve(A_2h, r_2h)  
        if G_VC.is_3d:
            np.save("A_2h", A_2h.todense())
    else: 
        e_init = np.zeros(G_VC.N)
        e2h, L, Nx_L = v_cicle(S1_it, S2_it, A_2h, r_2h, e_init, 
                              Grid_VC(G.N_x, G.N_y, G.N_z, G.L), Nx_L) 
                    
    # Prolongate and correct
    eh = prolongation(e2h, P, G_VC)
    un = correction(eh, u, G_VC)  
    uv2 = postsmoothing(S2_it, A, b, G_VC, un)   
   
    return uv2, G.L, Nx_L

# =============================================================================
# --------------------------- Main Multigrid Functions ----------------------
# =============================================================================

def MG(a, f, x_0, x_true, G, S1_it, S2_it, MG_it_max, tol, Comp_Eigs=False):
    """
    Multigrid V-cycle solver (unified 2D/3D)
    
    Parameters:
    a: system matrix
    f: right-hand side
    x_0: initial guess
    x_true: exact solution
    G: grid object
    S1_it: pre-smoothing iterations
    S2_it: post-smoothing iterations
    MG_it_max: maximum multigrid iterations
    tol: convergence tolerance
    Comp_Eigs: compute eigenvalues flag
    """
    L = 0
    Nx_L = [G.Nx + 1]
    t_In = 0
    t_It = 0 
    t_unit_f = 0
    t_In_p = 0
    t_It_p = 0
    
    # Initialization timing
    t_In_I = time.time()
    t_In_I_p = time.process_time()
 
    A, b, x, G_VC = init(a, f, x_0, G.Nx, G.Ny, G.Nz, 0)   
    rr = 1
    MG_it = 0
    
    # Compute initial residual
    r_h = b - A.dot(x)
    rr_vec = [] 
    e_vec = [] 
    rr = LA.norm(r_h) / LA.norm(b)
    rr_vec.append(rr) 
    Its_vec = []
    Its = 0
    Its_vec.append(Its) 
    
    # End initialization timing
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    t_In_F_p = time.process_time()
    t_In_p = t_In_I_p - t_In_F_p 
    
    re = LA.norm(x_true - x) / LA.norm(x_true)
    e_vec.append(re)
    
    # Start iteration timing
    t_It_I = time.time() 
    t_It_I_p = time.process_time()
    Its = 0
    flops_init, flops_iter = [0, 0]
    flops = 0

    # Main multigrid iteration loop
    while (rr > tol) and (MG_it < MG_it_max):
        if Its == 0:
            t_unit = time.time()
        G_VC.L = 0
        
        Nx_L = [G.Nx + 1]
        uv2, L, Nx_L = v_cicle(S1_it, S2_it, A, b, x, G_VC, Nx_L)  
        x = uv2.copy()
        r_h = b - A.dot(x)
        rr = LA.norm(r_h) / LA.norm(b)
        MG_it = MG_it + 1

        if Its == 0:
            t_unit_f = time.time() - t_unit
        rr_vec.append(rr)
        Its_vec.append(MG_it)
        re = LA.norm(x_true - x) / LA.norm(x_true)
        e_vec.append(re)
        Its += 1
        
        if Comp_Eigs:
            break
            
    # Calculate FLOPS
    flops_iter = MG_op(np.unique(Nx_L), S1_it, S2_it, m)
    flops += flops_iter.subs([(m, 5)])
        
    # End iteration timing
    t_It_F = time.time()
    t_It = t_It_I - t_It_F
    
    t_It_F_p = time.process_time()
    t_It_p = t_It_I_p - t_It_F_p
    
    # Prepare output
    t_vec = np.array([-t_In, -t_It, t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    G.L = L
    r_true = b - A.dot(x)
    MG_r = MGC(x, S1_it, S2_it, Its_vec, t_vec, t_vec_p, rr_vec, r_true, G, [0, flops], L, Nx_L, x_true, e_vec)

    return MG_r

def MGCG(a, b, x_0, x_true, G, MaxIter, tol, S1_it, S2_it):
    """
    Multigrid-preconditioned Conjugate Gradient method (unified 2D/3D)
    
    Parameters:
    a: system matrix
    b: right-hand side
    x_0: initial guess
    x_true: exact solution
    G: grid object
    MaxIter: maximum iterations
    tol: convergence tolerance
    S1_it: pre-smoothing iterations
    S2_it: post-smoothing iterations
    """
    L = 0
    Nx_L = [G.Nx + 1]
    t_In = 0
    t_It = 0 
    t_unit_f = 0
    t_In_p = 0
    t_It_p = 0
    
    # Initialization timing
    t_In_I = time.time()
    t_In_I_p = time.process_time()

    A = a.copy()
    b = b.copy()
    x = x_0.copy()
    r0 = 0 * x_0.copy()
    r = b - A.dot(x)
    G_VC = Grid_VC(G.Nx, G.Ny, G.Nz, 0)
    z = r0.copy()
    MG_it = 0
    
    # Initial multigrid solve for preconditioner
    while LA.norm(r - A*z) / LA.norm(r) > 0.05:  
        G_VC.L = 0
        Nx_L = [G.Nx]
        z, L, Nx_L = v_cicle(S1_it, S2_it, A, r, z, G_VC, Nx_L)

    p = z

    # Initialize convergence tracking
    rr_vec = [] 
    e_vec = [] 
    r = b - A.dot(x)
    rr = LA.norm(r) / LA.norm(b)
    rr_vec.append(rr) 

    Its_vec = []
    Its = 0
    Its_vec.append(Its)   
    
    # End initialization timing
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    re = LA.norm(x_true - x) / LA.norm(x_true)
    e_vec.append(re)
            
    t_In_F_p = time.process_time()
    t_In_p = t_In_I_p - t_In_F_p    
    
    # Start iteration timing
    t_It_I = time.time() 
    t_It_I_p = time.process_time()
    
    # Main MGCG iteration loop
    It_mg = []
    Nx_L = [G.Nx]
    while rr > tol and Its < MaxIter:
        x_old = x.copy()
        r_old = r.copy()
        z_old = z.copy()
        p_old = p.copy()
        
        if Its == 0:
            t_unit = time.time()
            
        w = A.dot(p_old)
        alpha = np.dot(z_old, r_old) / np.dot(w, p_old)
        x = x_old + alpha * p_old
        r = r_old - alpha * w
        
        # Multigrid preconditioning
        z = r0.copy()
        IT_MG = 0
        while LA.norm(r - A*z) / LA.norm(r) > 0.05:  
            G_VC.L = 0
            z, L, Nx_L = v_cicle(S1_it, S2_it, A, r, z, G_VC, Nx_L)
            IT_MG += 1
        It_mg.append(IT_MG)    
        
        beta = np.dot(z, r) / np.dot(z_old, r_old)
        p = z + beta * p_old 
        
        if Its == 0:
            t_unit_f = time.time() - t_unit
            
        rr = LA.norm(r) / LA.norm(b)
        re = LA.norm(x_true - x) / LA.norm(x_true)
        e_vec.append(re)

        Its += 1
        rr_vec.append(rr)
        Its_vec.append(Its) 

    # End iteration timing
    t_It_F = time.time()
    t_It = t_It_I - t_It_F
    
    t_It_F_p = time.process_time()
    t_It_p = t_It_I_p - t_It_F_p
    
    # Calculate FLOPS
    flops_init, flops_iter = CG_op(sv, vs, vv, Av_s, AB_s)
    flops_iter += np.mean(It_mg) * MG_op(np.unique(Nx_L), S1_it, S2_it, m)
    flops = np.array([flops_init.subs([(n, G.N), (m, 5)]), 
                     flops_iter.subs([(n, G.N), (m, 5)])])
    
    # Prepare output
    t_vec = np.array([-t_In, -t_It, t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    r_true = b - A.dot(x)
    
    MGCG_r = MGCGS(x, Its_vec, t_vec, t_vec_p, rr_vec, r_true, G, flops, L, Nx_L, x_true, e_vec)
    return MGCG_r

def MGCG1(a, b, x_0, x_true, G, MaxIter, tol, S1_it, S2_it):
    """
    Multigrid-preconditioned Conjugate Gradient method variant 1 (unified 2D/3D)
    
    Similar to MGCG but with single multigrid solve per iteration
    """
    L = 0
    Nx_L = [G.Nx + 1]
    t_In = 0
    t_It = 0 
    t_unit_f = 0
    t_In_p = 0
    t_It_p = 0
    
    # Initialization timing
    t_In_I = time.time()
    t_In_I_p = time.process_time()

    A = a.copy()
    b = b.copy()
    x = x_0.copy()
    r0 = 0 * x_0.copy()
    r = b - A.dot(x)
    G_VC = Grid_VC(G.Nx, G.Ny, G.Nz, 0)
    z = r0.copy()
    MG_it = 0
    
    # Single initial multigrid solve
    G_VC.L = 0
    Nx_L = [G.Nx]
    z, L, Nx_L = v_cicle(S1_it, S2_it, A, r, z, G_VC, Nx_L)

    p = z

    # Initialize convergence tracking
    rr_vec = [] 
    e_vec = [] 
    r = b - A.dot(x)
    rr = LA.norm(r) / LA.norm(b)
    rr_vec.append(rr) 

    Its_vec = []
    Its = 0
    Its_vec.append(Its)   
    
    # End initialization timing
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    re = LA.norm(x_true - x) / LA.norm(x_true)
    e_vec.append(re)
            
    t_In_F_p = time.process_time()
    t_In_p = t_In_I_p - t_In_F_p    
    
    # Start iteration timing
    t_It_I = time.time() 
    t_It_I_p = time.process_time()
    
    # Main MGCG1 iteration loop
    It_mg = []
    Nx_L = [G.Nx]
    while rr > tol and Its < MaxIter:
        x_old = x.copy()
        r_old = r.copy()
        z_old = z.copy()
        p_old = p.copy()
        
        if Its == 0:
            t_unit = time.time()
            
        w = A.dot(p_old)
        alpha = np.dot(z_old, r_old) / np.dot(w, p_old)
        x = x_old + alpha * p_old
        r = r_old - alpha * w
        
        # Single multigrid solve per iteration
        z = r0.copy()
        IT_MG = 0
        
        G_VC.L = 0
        z, L, Nx_L = v_cicle(S1_it, S2_it, A, r, z, G_VC, Nx_L)
        IT_MG += 1
        It_mg.append(IT_MG)    
        
        beta = np.dot(z, r) / np.dot(z_old, r_old)
        p = z + beta * p_old 
        
        if Its == 0:
            t_unit_f = time.time() - t_unit
            
        rr = LA.norm(r) / LA.norm(b)
        re = LA.norm(x_true - x) / LA.norm(x_true)
        e_vec.append(re)

        Its += 1
        rr_vec.append(rr)
        Its_vec.append(Its) 

    # End iteration timing
    t_It_F = time.time()
    t_It = t_It_I - t_It_F
    
    t_It_F_p = time.process_time()
    t_It_p = t_It_I_p - t_It_F_p
    
    # Calculate FLOPS
    flops_init, flops_iter = CG_op(sv, vs, vv, Av_s, AB_s)
    flops_iter += MG_op(np.unique(Nx_L), S1_it, S2_it, m)
    flops = np.array([flops_init.subs([(n, G.N), (m, 5)]), 
                     flops_iter.subs([(n, G.N), (m, 5)])])
    
    # Prepare output
    t_vec = np.array([-t_In, -t_It, t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    r_true = b - A.dot(x)
    
    MGCG_r = MGCGS1(x, Its_vec, t_vec, t_vec_p, rr_vec, r_true, G, flops, L, Nx_L, x_true, e_vec)
    return MGCG_r