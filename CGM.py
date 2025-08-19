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
import time
import numpy.linalg as LA
from scipy.sparse import linalg as SLA
import numpy.random
from scipy.sparse import csc_matrix, lil_matrix
from scipy import sparse
import sys
from def_func import *
from scipy.sparse import spdiags
from flops_op import *

# =============================================================================
# ----------------------------- Result Classes -------------------------------
# =============================================================================

class CGS:
    """Conjugate Gradient Solver results"""
    def __init__(self, x, Its, t, tp, rr, r_true, G, flops, x_true=[], re=[]):
        self.name = 'CG'
        self.marker = 'd'
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
        self.flops_init = float(flops[0])
        self.flops_iter = float(flops[1])

class PCGS:
    """Preconditioned Conjugate Gradient Solver results"""
    def __init__(self, x, Its, t, tp, rr, r_true, G, flops, pre, x_true=[], re=[]):
        self.name = pre + 'CG'
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
        self.flops_init = float(flops[0])
        self.flops_iter = float(flops[1])

class DPCGS:
    """Deflated Preconditioned Conjugate Gradient Solver results"""
    def __init__(self, x, Its, tu, t, tp, rr, r_true, G, flops, pre, dv, x_true=[], re=[]):
        self.name = 'D' + pre + 'CG'
        self.marker = '<'
        self.x = x
        self.t_unit = tu
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
        self.flops_init = float(flops[0])
        self.flops_iter = float(flops[1])

class DCGS:
    """Deflated Conjugate Gradient Solver results"""
    def __init__(self, x, Its, t, tp, rr, r_true, G, flops, dv, x_true=[], re=[]):
        self.name = 'DCG'
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
        self.flops_init = float(flops[0])
        self.flops_iter = float(flops[1])

# =============================================================================
# --------------------------- Solver Functions -------------------------------
# =============================================================================

def CG(a, b, x_0, x_true, G, MaxIter, tol):
    """
    Conjugate Gradient method
    
    Parameters:
    a: coefficient matrix
    b: right-hand side vector
    x_0: initial guess
    x_true: exact solution
    G: grid object
    MaxIter: maximum iterations
    tol: tolerance
    """
    # Initialize timing variables
    t_In = 0
    t_It = 0 
    t_unit_f = 0
    t_In_p = 0
    t_It_p = 0
    
    # Start initialization timing
    t_In_I = time.time()  
    t_In_I_p = time.process_time()

    # Copy input arrays
    A = a.copy()
    b = b.copy()
    x = x_0.copy()
    
    # Initial residual and search direction
    r = b - A.dot(x)
    p = r.copy()
    
    # Initialize result vectors
    rr_vec = [] 
    r_true = r
    rr = LA.norm(r_true)/LA.norm(b)
    rr_vec.append(rr) 
    
    e_vec = []  
    re = LA.norm(x_true-x)/LA.norm(x_true)
    e_vec.append(re)
    
    Its_vec = []
    Its = 0
    Its_vec.append(Its)   
    
    # End initialization timing
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    
    t_In_F_p = time.process_time()
    t_In_p = t_In_I_p - t_In_F_p    
    
    # Start iteration timing
    t_It_I = time.time() 
    t_It_I_p = time.process_time()
    
    # Main CG iteration loop
    while rr > tol and Its < MaxIter:
        x_old = x.copy()
        r_old = r.copy()
        p_old = p.copy()
        
        if Its == 0:
            t_unit = time.time()
            
        w = A.dot(p_old)
        alpha = np.dot(r_old, r_old) / np.dot(w, p_old)
        x = x_old + alpha * p_old
        r = r_old - alpha * w   
        beta = np.dot(r, r) / np.dot(r_old, r_old)
        p = r + beta * p_old 
        
        if Its == 0:
            t_unit_f = time.time() - t_unit
            
        # Update residual and error
        r = b - A.dot(x)
        rr = LA.norm(r) / LA.norm(b)
        Its += 1
        rr_vec.append(rr)
        Its_vec.append(Its)
        
        x_true = SLA.spsolve(A, b)
        re = LA.norm(x_true - x) / LA.norm(x_true)
        e_vec.append(re)
            
    # End iteration timing
    t_It_F = time.time()
    t_It = t_It_I - t_It_F
    
    t_It_F_p = time.process_time()
    t_It_p = t_It_I_p - t_It_F_p
    
    # Calculate FLOPS
    flops_init, flops_iter = CG_op(sv, vs, vv, Av_s, AB_s)
    flops = np.array([flops_init.subs([(n, G.N), (m, 5)]), 
                     flops_iter.subs([(n, G.N), (m, 5)])])

    # Prepare output
    t_vec = np.array([-t_In, -t_It, t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    r_true = b - A.dot(x)

    CG_r = CGS(x, Its_vec, t_vec, t_vec_p, rr_vec, r_true, G, flops, x_true, e_vec)
    return CG_r

def PGC(a, b, x_0, x_true, G, MaxIter, pre, tol):
    """
    Preconditioned Conjugate Gradient method
    
    Parameters:
    a: coefficient matrix
    b: right-hand side vector
    x_0: initial guess
    x_true: exact solution
    G: grid object
    MaxIter: maximum iterations
    pre: preconditioner type
    tol: tolerance
    """
    # Initialize timing variables
    t_In = 0
    t_It = 0 
    t_unit_f = 0
    t_In_p = 0
    t_It_p = 0
    
    # Start initialization timing
    t_In_I = time.time()
    t_In_I_p = time.process_time()
    
    A = a.copy()
    b = b.copy()
    x = x_0.copy()

    # Jacobi preconditioner
    d = 1/A.diagonal()
    M1 = spdiags(d, 0, G.N, G.N)
    r = b - A.dot(x)

    y = M1 * r
    ry = np.dot(r, y)
    p = y

    Mb = M1 * b

    # Initialize result vectors
    rr_vec = [] 
    r_true = r
    rr = LA.norm(r) / LA.norm(b)
    rr_vec.append(rr) 
    
    e_vec = []  
    re = LA.norm(x_true - x) / LA.norm(x_true)
    e_vec.append(re)
    
    Its_vec = []
    Its = 0
    Its_vec.append(Its)   
    
    # End initialization timing
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    
    t_In_F_p = time.process_time()
    t_In_p = t_In_I_p - t_In_F_p    
    
    # Start iteration timing
    t_It_I = time.time() 
    t_It_I_p = time.process_time()
    
    # Main PCG iteration loop
    while rr > tol and Its < MaxIter:
        x_old = x.copy()
        r_old = r.copy()
        y_old = y.copy()
        ry_old = ry.copy()
        p_old = p.copy()
        
        if Its == 0:
            t_unit = time.time()
            
        w = A.dot(p_old)
        alpha = ry_old / np.dot(p_old, w)
        x = x_old + alpha * p_old
        r = r_old - alpha * w
        y = M1 * r
        ry = np.dot(r, y)
        beta = ry / ry_old
        p = y + beta * p_old 
        
        if Its == 0:
            t_unit_f = time.time() - t_unit

        r = b - A.dot(x)
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
    
    # Calculate FLOPS (simplified)
    flops = np.array([0, 1])
    
    # Prepare output
    t_vec = np.array([-t_In, -t_It, t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    r_true = b - A.dot(x)
    
    PCG_r = PCGS(x, Its_vec, t_vec, t_vec_p, rr_vec, r_true, G, flops, pre, x_true, e_vec)
    return PCG_r

def DPGC(a, b, x_0, x_true, G, MaxIter, pre, tol, dv):
    """
    Deflated Preconditioned Conjugate Gradient method
    
    Parameters:
    a: coefficient matrix
    b: right-hand side vector
    x_0: initial guess
    x_true: exact solution
    G: grid object
    MaxIter: maximum iterations
    pre: preconditioner type
    tol: tolerance
    dv: deflation vector type
    """
    # Initialize timing variables
    t_In = 0
    t_It = 0 
    t_unit_f = 0
    t_In_p = 0
    t_It_p = 0
    
    # Start initialization timing
    t_In_I = time.time()
    t_In_I_p = time.process_time()
    
    A = a.copy()
    b = b.copy()
    x = x_0.copy()
    
    # Jacobi preconditioner
    d = 1/A.diagonal()
    M1 = spdiags(d, 0, G.N, G.N)

    # Deflation setup
    Z = G.Z
    ZT = Z.transpose()
    V = A * Z
    V = csc_matrix(V)
    E = Z.transpose() * V
    E = csc_matrix(E)
    EI = SLA.inv(E)   
    B = V * EI
    B = csc_matrix(B)
    del d, V, E
    
    # Initialize variables
    r = b - A.dot(x)
    rr = LA.norm(r) / LA.norm(b)
    r = Deflation_P(B, Z, r)
    y = M1 * r
    p_n = y
    Pb = Deflation_P(B, Z, b)

    # Initialize result vectors
    rr_vec = [] 
    rr_vec.append(rr) 

    e_vec = []  
    re = LA.norm(x_true - x) / LA.norm(x_true)
    e_vec.append(re)
    
    Its_vec = []
    Its = 0
    x_it = 0
    Its_vec.append(Its)   
    ry = np.dot(r, y)
    
    # End initialization timing
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    
    t_In_F_p = time.process_time()
    t_In_p = t_In_I_p - t_In_F_p    
    
    # Start iteration timing
    t_It_I = time.time() 
    t_It_I_p = time.process_time()
    
    # Main DPCG iteration loop
    while rr > tol and Its < MaxIter:
        x_old = x.copy()
        r_old = r.copy()
        y_old = y.copy()
        p_old = p_n.copy()
        ry_old = ry.copy()
        
        if Its == 0:
            t_unit = time.time()
            
        w = A * p_old
        w = w - B.dot(ZT.dot(w))
        alpha = ry_old / np.dot(p_old, w)
        x = x_old + alpha * p_old
        r = r_old - alpha * w
        y = M1 * r
        ry = np.dot(r, y)
        beta = ry / ry_old
        p_n = y + beta * p_old 
        
        if Its == 0:
            t_unit_f = time.time() - t_unit

        x_it = Correction_Q(Z, EI, b) + Deflation_P(B, Z, x, T=True)
        r = b - A.dot(x_it)
        rr = LA.norm(r) / LA.norm(b)
        re = LA.norm(x_true - x_it) / LA.norm(x_true)
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
    flops_init, flops_iter = DPCG_op(sv, vs, vv, Av_s, AB_s, AZ_s, VZ_p, InvE, VE_p, Br_p, Zv_p)
    flops = np.array([flops_init.subs([(n, G.N), (m, 5), (p, np.shape(G.Z)[1])]), 
                     flops_iter.subs([(n, G.N), (m, 5), (p, np.shape(G.Z)[1])])])
    
    # Prepare output
    t_vec = np.array([-t_In, -t_It])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    x_it = Correction_Q(Z, EI, b) + Deflation_P(B, Z, x, T=True)
    r_true = b - A.dot(x_it)
   
    DPCG_r = DPCGS(x_it, Its_vec, t_unit_f, t_vec, t_vec_p, rr_vec, r_true, G, flops, pre, dv, x_true, e_vec)
    return DPCG_r

def DGC(a, b, x_0, x_true, G, MaxIter, tol, dv):
    """
    Deflated Conjugate Gradient method
    
    Parameters:
    a: coefficient matrix
    b: right-hand side vector
    x_0: initial guess
    x_true: exact solution
    G: grid object
    MaxIter: maximum iterations
    tol: tolerance
    dv: deflation vector type
    """
    # Initialize timing variables
    t_In = 0
    t_It = 0 
    t_unit_f = 0
    t_In_p = 0
    t_It_p = 0
    
    # Start initialization timing
    t_In_I = time.time()
    t_In_I_p = time.process_time()
    
    A = a.copy()
    b = b.copy()
    x = x_0.copy()
    x_it = x_0.copy
    
    # Deflation setup
    Z = G.Z
    ZT = Z.T
    V = A * Z
    V = csc_matrix(V)
    E = Z.transpose() * V
    E = csc_matrix(E)
    EI = SLA.inv(E)
    B = V * EI
    B = csc_matrix(B)
    
    # Initialize variables
    r = b - A.dot(x)
    rr = LA.norm(r) / LA.norm(b)
    r = Deflation_P(B, Z, r)
    Pb = Deflation_P(B, Z, b)
    p_n = r.copy()

    # Initialize result vectors
    rr_vec = [] 
    rr_vec.append(rr) 
    
    e_vec = []  
    re = LA.norm(x_true - x) / LA.norm(x_true)
    e_vec.append(re)
    
    Its_vec = []
    Its = 0
    x_it = 0
    Its_vec.append(Its)   
    
    # End initialization timing
    t_In_F = time.time()
    t_In = t_In_I - t_In_F
    
    t_In_F_p = time.process_time()
    t_In_p = t_In_I_p - t_In_F_p    
    
    # Start iteration timing
    t_It_I = time.time() 
    t_It_I_p = time.process_time()
    
    # Main DCG iteration loop
    while rr > tol and Its < MaxIter:
        x_old = x.copy()
        r_old = r.copy()
        p_old = p_n.copy()
     
        if Its == 0:
            t_unit = time.time() 
            
        w = A * p_old
        w = w - B.dot(ZT.dot(w))
        alpha = np.dot(r_old, r_old) / np.dot(w, p_old)
        x = x_old + alpha * p_old
        r = r_old - alpha * w
        beta = np.dot(r, r) / np.dot(r_old, r_old)
        p_n = r + beta * p_old 
        
        if Its == 0:
            t_unit_f = time.time() - t_unit

        x_it = Correction_Q(Z, EI, b) + Deflation_P(B, Z, x, T=True)
        r = b - A.dot(x_it)
        rr = LA.norm(r) / LA.norm(b)
        re = LA.norm(x_true - x_it) / LA.norm(x_true)
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
    flops_init, flops_iter = DCG_op(sv, vs, vv, Av_s, AB_s, AZ_s, VZ_p, InvE, VE_p, Br_p, Zv_p)
    flops = np.array([flops_init.subs([(n, G.N), (m, 5), (p, np.shape(G.Z)[1])]), 
                     flops_iter.subs([(n, G.N), (m, 5), (p, np.shape(G.Z)[1])])])
    
    # Prepare output
    t_vec = np.array([-t_In, -t_It, t_unit_f])
    t_vec_p = np.array([-t_In_p, -t_It_p])
    x_it = Correction_Q(Z, EI, b) + Deflation_P(B, Z, x, T=True)
    r_true = b - A.dot(x_it)

    DCG_r = DCGS(x_it, Its_vec, t_vec, t_vec_p, rr_vec, r_true, G, flops, dv, x_true, e_vec)
    return DCG_r