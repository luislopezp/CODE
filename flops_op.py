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
import sympy as sym
import numpy as np

# Symbolic variables
m = sym.Symbol('m')  # Number of non-zero diagonals in sparse matrix
n = sym.Symbol('n')  # Problem size (number of unknowns)
p = sym.Symbol('p')  # Number of deflation vectors

# =============================================================================
# --------------------------- Basic Operations -------------------------------
# =============================================================================

# Vector operations
norm_v = 2*n                # Euclidean norm of a vector
v_inv = n                   # Invert each element of diagonal
M_1_v = n                   # Apply diagonal preconditioner
sv = n                      # Scalar-vector multiplication
vs = n                      # Vector addition/subtraction
vve = n                     # Vector-vector element-wise operations
vv = 2*n - 1               # Vector dot product

# Matrix-vector operations
Av_f = (2*n - 1)*n         # Full matrix-vector multiplication
Av_s = (2*m - 1)*n         # Sparse matrix-vector multiplication

# Matrix-matrix operations
AB_f = 2*n**3 - n**2       # Full matrix-matrix multiplication
AB_s_add = n*m             # Sparse matrix addition
AB_s = (2*m - 1)*n**2      # Sparse matrix-matrix multiplication

# Deflation-specific operations
AZ_s = (2*m - 1)*n*p       # A(sparse) * Z(sparse) multiplication
VZ_p = (2*n - 1)*p*p       # V(sparse) * Z(sparse) multiplication
VE_p = (2*p - 1)*n*p       # V(sparse) * E multiplication
Br_p = (2*n - 1)*p         # B * r multiplication
Zv_p = (2*p - 1)*n         # Z * v multiplication

# Matrix decomposition
InvE = 2*p**3/3            # Matrix inversion (p×p)
ChF = n**3/3               # Cholesky decomposition
FS = n**2                  # Forward substitution
BS = n**2                  # Backward substitution

# =============================================================================
# --------------------------- Cholesky Method --------------------------------
# =============================================================================

def Chol_op(ChF, FS, BS):
    """
    Calculate FLOPS for Cholesky decomposition method
    
    Parameters:
    ChF: Cholesky factorization operations
    FS: Forward substitution operations
    BS: Backward substitution operations
    
    Returns:
    flops_init: Initialization FLOPS
    flops_iter: Per-iteration FLOPS
    """
    # Initialization
    flops_init = 0
    
    # Iteration process
    LLT = ChF        # A = LL^T
    y = FS           # y = L^(-1) * b
    x = BS           # x = (L^T)^(-1) * y
    flops_iter = LLT + y + x
    
    return sym.simplify(flops_init), sym.simplify(flops_iter)

# =============================================================================
# --------------------------- Conjugate Gradient -----------------------------
# =============================================================================

def CG_op(sv, vs, vv, Av_s, AB_s):
    """
    Calculate FLOPS for Conjugate Gradient method
    
    Parameters:
    sv: Scalar-vector multiplication
    vs: Vector addition/subtraction
    vv: Vector dot product
    Av_s: Sparse matrix-vector multiplication
    AB_s: Sparse matrix-matrix multiplication
    
    Returns:
    flops_init: Initialization FLOPS
    flops_iter: Per-iteration FLOPS
    """
    # Initialization
    r0 = Av_s + vs    # r0 = b - A*x0
    flops_init = r0
    
    # Iteration process
    w = Av_s         # w = A*p
    rr = vv          # (r_j, r_j)
    alpha = rr + vv + 1  # alpha = (r_j, r_j) / (w_j, p_j)
    x = sv + vs      # x_{j+1} = x_j + alpha*p_j
    r = sv + vs      # r_{j+1} = r_j - alpha*w_j
    beta = vv + 1    # beta = (r_{j+1}, r_{j+1}) / (r_j, r_j)
    p = sv + vs      # p_{j+1} = r_{j+1} + beta*p_j
    flops_iter = w + rr + alpha + x + r + beta + p
    
    return sym.simplify(flops_init), sym.simplify(flops_iter)

# =============================================================================
# ------------------------ Preconditioned Conjugate Gradient ----------------
# =============================================================================

def PCG_op(v_inv, sv, vs, vv, Av_s, norm_v, M_1_v):
    """
    Calculate FLOPS for Preconditioned Conjugate Gradient method
    
    Parameters:
    v_inv: Vector inversion
    sv: Scalar-vector multiplication
    vs: Vector addition/subtraction
    vv: Vector dot product
    Av_s: Sparse matrix-vector multiplication
    norm_v: Vector norm
    M_1_v: Preconditioner application
    
    Returns:
    flops_init: Initialization FLOPS
    flops_iter: Per-iteration FLOPS
    """
    # Initialization
    M_1 = v_inv        # M^(-1) = 1/diag(A)
    r0 = Av_s + vs     # r0 = b - A*x0
    y0 = M_1_v         # y0 = M^(-1)*r0
    ry = vv            # (r, y)
    norm_r = norm_v    # ||r||
    norm_b = norm_v    # ||b||
    rr = 1             # ||r|| / ||b||
    
    flops_init = v_inv + r0 + y0 + ry + norm_r + norm_b + rr
    
    # Iteration process
    w = Av_s         # w = A*p
    rr = vv          # (w_j, p_j)
    alpha = 1        # alpha = ry / rr
    x = sv + vs      # x_{j+1} = x_j + alpha*p_j
    r = sv + vs      # r_{j+1} = r_j - alpha*w_j
    y = M_1_v        # y = M^(-1)*r
    ry = vv          # (r_{j+1}, y_{j+1})
    beta = 1         # beta = ry / rr
    p = sv + vs      # p_{j+1} = y_{j+1} + beta*p_j
    r1 = sv + Av_s   # r_{j+1} = b - A*x_{j+1}
    norm_r = norm_v  # ||r||
    norm_b = norm_v  # ||b||
    rr1 = 1          # ||r|| / ||b||
    
    flops_iter = w + rr + alpha + x + r + y + ry + beta + p + r1 + norm_r + norm_b + rr1
    
    return sym.simplify(flops_init), sym.simplify(flops_iter)

# =============================================================================
# --------------------------- Deflated Conjugate Gradient -------------------
# =============================================================================

def DCG_op(sv, vs, vv, Av_s, AZ_s, VZ_p, InvE, VE_p, Br_p, Zv_p, norm_v):
    """
    Calculate FLOPS for Deflated Conjugate Gradient method
    
    Parameters:
    sv: Scalar-vector multiplication
    vs: Vector addition/subtraction
    vv: Vector dot product
    Av_s: Sparse matrix-vector multiplication
    AZ_s: A*Z multiplication
    VZ_p: V*Z multiplication
    InvE: Matrix inversion
    VE_p: V*E multiplication
    Br_p: B*r multiplication
    Zv_p: Z*v multiplication
    norm_v: Vector norm
    
    Returns:
    flops_init: Initialization FLOPS
    flops_iter: Per-iteration FLOPS
    """
    # Initialization
    V = AZ_s          # V = A*Z
    E = VZ_p          # E = Z^T*V
    EI = InvE         # EI = E^(-1)
    B = VE_p          # B = V*EI
    
    r0 = Av_s + vs    # r0 = b - A*x0
    norm_r = norm_v   # ||r||
    norm_b = norm_v   # ||b||
    rr = 1            # ||r|| / ||b||
    P = Br_p + Zv_p + vs     # Deflation_P(B, Z, r)
    M_vp = (2*p - 1)*n       # Z*v multiplication
    M_vpp = (2*p - 1)*p      # E*v multiplication  
    M_vn = (2*n - 1)*p       # B*r multiplication
    Q = M_vp + M_vpp + M_vn  # Correction_Q operations
    flops_init = r0 + V + E + EI + B + P + norm_r + norm_b + rr
    
    # Iteration process
    w = Av_s         # w = A*p
    Pw = Av_s + P    # Deflation_P(B, Z, w)
    rr = vv          # (r_j, r_j)
    wp = vv          # (w_j, p_j)
    alpha = 1        # alpha = (r_j, r_j) / (w_j, p_j)
    x = sv + vs      # x_{j+1} = x_j + alpha*p_j
    r = sv + vs      # r_{j+1} = r_j - alpha*w_j
    beta = vv + 1    # beta = (r_{j+1}, r_{j+1}) / (r_j, r_j)
    p_update = sv + vs      # p_{j+1} = r_{j+1} + beta*p_j
    x_it = Q + P + sv        # Correction and deflation
    r_it = Av_s + vs # r_it = b - A*x_it
    norm_r = norm_v  # ||r||
    norm_b = norm_v  # ||b||
    rr_final = 1     # ||r|| / ||b||
    
    flops_iter = w + Pw + rr + wp + alpha + x + r + beta + p_update + x_it + r_it + norm_r + norm_b + rr_final
    
    return sym.simplify(flops_init), sym.simplify(flops_iter)

# =============================================================================
# -------------------- Deflated Preconditioned Conjugate Gradient -----------
# =============================================================================

def DPCG_op(sv, vs, vv, Av_s, AB_s, AZ_s, VZ_p, InvE, VE_p, Br_p, Zv_p):
    """
    Calculate FLOPS for Deflated Preconditioned Conjugate Gradient method
    
    Parameters:
    sv: Scalar-vector multiplication
    vs: Vector addition/subtraction
    vv: Vector dot product
    Av_s: Sparse matrix-vector multiplication
    AB_s: Sparse matrix-matrix multiplication
    AZ_s: A*Z multiplication
    VZ_p: V*Z multiplication
    InvE: Matrix inversion
    VE_p: V*E multiplication
    Br_p: B*r multiplication
    Zv_p: Z*v multiplication
    
    Returns:
    flops_init: Initialization FLOPS
    flops_iter: Per-iteration FLOPS
    """
    # Initialization
    V = AZ_s          # V = A*Z
    E = VZ_p          # E = Z^T*V
    EI = InvE         # EI = E^(-1)
    B = VE_p          # B = V*EI
    
    r0 = Av_s + vs    # r0 = b - A*x0
    P = Br_p + Zv_p + vs     # Deflation_P(B, Z, r)
    y0 = Av_s         # y0 = M^(-1)*r0
    flops_init = r0 + V + E + EI + B + P + y0
    
    # Iteration process
    Pw = Av_s + P     # w = P*A*p
    ry = vv           # (r_j, y_j)
    alpha = ry + vv + 1  # alpha = (r_j, y_j) / (w_j, p_j)
    x = sv + vs       # x_{j+1} = x_j + alpha*p_j
    r = sv + vs       # r_{j+1} = r_j - alpha*w_j
    y = vve           # y = M^(-1)*r
    beta = vv + 1     # beta = (r_{j+1}, y_{j+1}) / (r_j, y_j)
    p = sv + vs       # p_{j+1} = y_{j+1} + beta*p_j
    
    flops_iter = Pw + ry + alpha + x + r + y + beta + p
    
    return sym.simplify(flops_init), sym.simplify(flops_iter)

# =============================================================================
# --------------------------- Multigrid Operations ---------------------------
# =============================================================================

def Av_sf(n):
    """Sparse matrix-vector multiplication for multigrid"""
    return 2*m*n

def Rr_n(N, n):
    """Restriction operation"""
    return (2*n - 1)*N

def RAP_n(N, n):
    """Galerkin coarse grid operator"""
    return (2*m - 1)*(N*N + n*N)

def PeH_n(N, n):
    """Prolongation operation"""
    return (2*N - 1)*n

def xh_n(N):
    """Grid correction"""
    return N

def ps_n(m, S_it, N):
    """Pre/post smoothing operations"""
    return (2*m*S_it + 4)*N

def MG_op(Nx_L, S1_it, S2_it, m):
    """
    Calculate FLOPS for Multigrid V-cycle
    
    Parameters:
    Nx_L: Grid sizes at each level
    S1_it: Pre-smoothing iterations
    S2_it: Post-smoothing iterations
    m: Number of non-zero diagonals
    
    Returns:
    flops_iter: Per-iteration FLOPS
    """
    flops_init = 0  # Compute relative residual
    
    # Pre-smoothing
    pS1 = np.sum([ps_n(m, S1_it, n) for n in Nx_L])
    
    # Residual computation
    rh = np.sum([Av_sf(n) for n in Nx_L])
    
    # Restriction
    Rrh = np.sum([Rr_n(Nx_L[i], Nx_L[j]) for i in np.arange(0, len(Nx_L)-1) for j in [i+1]])
    
    # Coarsest grid solve
    Solve_LS = Nx_L[-1]**3
    
    # Prolongation
    PeH = np.sum([PeH_n(Nx_L[i], Nx_L[j]) for i in np.arange(0, len(Nx_L)-1) for j in [i+1]])
    
    # Correction
    xh = np.sum([xh_n(n) for n in Nx_L])
    
    # Post-smoothing
    pS2 = np.sum([ps_n(m, S2_it, n) for n in Nx_L])
    
    flops_iter = pS1 + rh + Rrh + PeH + xh + Solve_LS + pS2
    
    return sym.simplify(flops_iter)

# =============================================================================
# --------------------------- Usage Examples ---------------------------------
# =============================================================================

# Generate FLOPS expressions for each method
CG_init, CG_iter = CG_op(sv, vs, vv, Av_s, AB_s)
PCG_init, PCG_iter = PCG_op(v_inv, sv, vs, vv, Av_s, norm_v, M_1_v)
DCG_init, DCG_iter = DCG_op(sv, vs, vv, Av_s, AZ_s, VZ_p, InvE, VE_p, Br_p, Zv_p, norm_v)
DPCG_init, DPCG_iter = DPCG_op(sv, vs, vv, Av_s, AB_s, AZ_s, VZ_p, InvE, VE_p, Br_p, Zv_p)
Chol_init, Chol_iter = Chol_op(ChF, FS, BS)