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
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.linalg as LA
import time
from scipy.sparse import lil_matrix, csr_matrix
from scipy import stats


# =============================================================================
# ------------------------- Utility Functions --------------------------------
# =============================================================================

def vec_to_mat(p, Nx, Ny):
    """
    Convert vector to matrix format
    
    Parameters:
    p: input vector
    Nx, Ny: grid dimensions
    """
    ap = np.zeros((Nx, Ny))
    for j in range(0, Ny):
        for i in range(0, Nx):
            h = i + j * Nx
            ap[i, j] = p[h]
    return ap

def mat_to_vect(A, Nx, Ny):
    """
    Convert matrix to vector format
    
    Parameters:
    A: input matrix
    Nx, Ny: grid dimensions
    """
    vA = np.zeros(Nx * Ny)
    for j in range(0, Ny):
        for i in range(0, Nx):
            h = i + j * Nx
            vA[h] = A[i, j]
    return vA

def mat_to_vect_new(A):
    """
    Convert 2D array to vector (alternative implementation)
    
    Parameters:
    A: input 2D array
    """
    col = A.shape[1]
    row = A.shape[0]
    vA = np.zeros(col * row)
    for i in range(0, row):
        for j in range(0, col):       
            h = col * i + j
            vA[h] = A[i, j]
    return vA

def Tens_to_vect_new(A):
    """
    Convert 3D tensor to vector
    
    Parameters:
    A: input 3D tensor
    """
    layers = A.shape[0]
    row = A.shape[1]
    col = A.shape[2]
    Np = col * row
    vA = np.zeros(col * row * layers)
    for k in range(0, layers):
        for i in range(0, row):
            for j in range(0, col):       
                h = col * i + j + Np * k
                vA[h] = A[k, i, j]
    return vA

# =============================================================================
# ------------------------ Averaging and Transmissibility --------------------
# =============================================================================

def lamda_av3D(y, G):
    """
    Calculate harmonic averages for 3D transmissibility
    
    Parameters:
    y: property values (permeability/viscosity)
    G: grid object
    """
    yx_av = np.zeros(G.N)
    yy_av = np.zeros(G.N)
    yz_av = np.zeros(G.N)
    
    # X-direction harmonic average
    yx_av[0:G.N-1] = 2 / ((1/y[0:G.N-1]) + (1/y[1:G.N])) 
    # Y-direction harmonic average
    yy_av[0:G.N-G.Nx] = 2 / ((1/y[0:G.N-G.Nx]) + (1/y[G.Nx:G.N]))
    # Z-direction harmonic average
    yz_av[0:G.N-G.Nl] = 2 / ((1/y[0:G.N-G.Nl]) + (1/y[G.Nl:G.N])) 
    
    return yx_av, yy_av, yz_av

def Trans_FV3D(yx_av, yy_av, yz_av, G):    
    """
    Calculate transmissibilities for 3D finite volume method
    
    Parameters:
    yx_av, yy_av, yz_av: averaged properties
    G: grid object
    """
    Tx = G.Ax * yx_av / G.dxc
    Ty = G.Ay * yy_av / G.dyc
    Tz = G.Az * yz_av / G.dzc
    
    return Tx, Ty, Tz

# =============================================================================
# ------------------------ Permeability Field Generation --------------------
# =============================================================================

def Hom(k_0, G):
    """
    Generate homogeneous permeability field
    
    Parameters:
    k_0: permeability value
    G: grid object
    """
    k = k_0 * np.ones(G.N)
    Nlayers = 4
    slice_lay = int(G.Ny / Nlayers)
    Z = lil_matrix((G.N, 4))
    Z[0*G.Nx*slice_lay : G.Nx*slice_lay, 0] = 1
    Z[G.Nx*slice_lay : G.Nx*slice_lay*(2), 1] = 1
    Z[2*G.Nx*slice_lay : G.Nx*slice_lay*(3), 2] = 1       
    Z[3*G.Nx*slice_lay:, 3] = 1
    return k, Z

def Lfrac_new_dist(kh, kl, G, Nfrac):
    """
    Generate distributed fracture permeability field with lognormal distribution
    
    Parameters:
    kh: high permeability value
    kl: low permeability value
    G: grid object
    Nfrac: number of fractures
    """
    # Low permeability lognormal distribution
    muL = kl
    sigmaL = 0.1 * kl
    muL_log = np.log(muL**2 / (np.sqrt(muL**2 + sigmaL**2)))
    sigmaL_log = np.sqrt(np.log(1 + sigmaL**2 / muL**2))
    kperm = np.random.lognormal(muL_log, sigmaL_log, G.N)
    
    # High permeability lognormal distribution
    muH = kh
    sigmaH = 0.1 * muH
    muH_log = np.log(muH**2 / (np.sqrt(muH**2 + sigmaH**2)))
    sigmaH_log = np.sqrt(np.log(1 + sigmaH**2 / muH**2))
    kH = np.random.lognormal(muH_log, sigmaH_log, G.N)

    # Create deflation matrix
    colz = 2 * Nfrac
    Z = lil_matrix((G.N, colz)) 
    
    # Assign high permeability to fracture locations
    for i in range(0, G.N):
        if i % (G.Nx * (G.Ny / Nfrac)) == 0:
            kperm[i:i + G.Nx] = kH[i:i + G.Nx]
    
    # Set up deflation vectors for fractures
    hf = np.zeros(Nfrac)
    hc = 0
    for i in range(0, G.N):   
        if i % (G.Nx * (G.Ny / Nfrac)) == 0:
            hf[hc] = i
            hc = hc + 1
    
    # Even indices for high permeability regions
    jpar = np.arange(0, Nfrac * 2, 2)
    for j in range(0, Nfrac):
        Z[int(hf[j]):int(hf[j]) + G.Nx, jpar[j]] = 1
    
    # Odd indices for low permeability regions
    hl = hf + G.Nx
    jimpar = np.arange(1, Nfrac * 2, 2)
    for j in range(0, Nfrac):
        Z[int(hl[j]):int(hl[j]) + int((G.Ny / Nfrac - 1) * G.Nx), jimpar[j]] = 1
    
    return kperm, Z

def Perm(G, case, perm_coef, Nfrac, lay=[], v=[], Clay=[]):
    """
    Generate permeability field based on case type
    
    Parameters:
    G: grid object
    case: case type ('Hom' for homogeneous, 'D_frac' for distributed fractures)
    perm_coef: permeability coefficients [high, low]
    Nfrac: number of fractures
    lay: layers (unused for current cases)
    v: additional parameters (unused)
    Clay: clay parameters (unused)
    """
    k_0 = perm_coef[0]  # High permeability
    k_1 = perm_coef[1]  # Low permeability
    
    # Homogeneous case
    if case == 'Hom':
        k, Z = Hom(k_0, G)
        
    # Distributed fractures case
    elif case == 'D_frac':
        if G.Nz == 1:
            k, Z = Lfrac_new_dist(k_0, k_1, G, Nfrac)
        else:
            # 3D implementation would go here
            k, Z = Lfrac_new_dist(k_0, k_1, G, Nfrac)
    else:
        raise ValueError(f"Case '{case}' not implemented in simplified version")
    
    return k, Z

# =============================================================================
# ------------------------ Matrix Assembly Functions ------------------------
# =============================================================================

def A_mat_FV_no_for_allB3D(Tx, Ty, Tz, G, y, bc):
    """
    Assemble finite volume matrix for 3D reservoir simulation
    
    Parameters:
    Tx, Ty, Tz: transmissibilities in x, y, z directions
    G: grid object
    y: property values
    bc: boundary conditions
    """
    # Initialize timing
    t_0 = time.time()

    # Pre-allocate matrix
    A = lil_matrix((G.N, G.N))
    
    # Create index arrays for neighbors and boundaries
    idx = np.arange(0, G.N)
    idxLN = np.array(np.where((idx % G.Nx) != 0))        # Left neighbors
    idxRN = np.array(np.where((idx % G.Nx) != G.Nx-1))   # Right neighbors
    idxNN = np.array(np.where(idx % G.Nl < G.Nl-G.Nx))   # North neighbors
    idxSN = np.array(np.where(idx % G.Nl > G.Nx-1))      # South neighbors
    idxUN = np.array(np.where(idx < G.N-G.Nl))           # Upper neighbors
    idxDN = np.array(np.where(idx > G.Nl-1))             # Down neighbors
    
    idxLB = np.array(np.where((idx % G.Nx) == 0))        # Left boundary
    idxRB = np.array(np.where((idx % G.Nx) == G.Nx-1))   # Right boundary
    idxNB = np.array(np.where(idx % G.Nl >= G.Nl-G.Nx))  # North boundary
    idxSB = np.array(np.where(idx % G.Nl <= G.Nx-1))     # South boundary
    idxUB = np.array(np.where(idx >= G.N-G.Nl))          # Upper boundary
    idxDB = np.array(np.where(idx <= G.Nl-1))            # Down boundary
    
    # Assemble interior connections
    # Left neighbors
    A[idxLN, idxLN] = Tx[idxLN-1]
    A[idxLN, idxLN-1] = -Tx[idxLN-1]
    
    # Right neighbors  
    A[idxRN, idxRN] += Tx[idxRN]
    A[idxRN, idxRN+1] -= Tx[idxRN]
    
    # North neighbors
    A[idxNN, idxNN] += Ty[idxNN]
    A[idxNN, idxNN+G.Nx] -= Ty[idxNN]
    
    # South neighbors
    A[idxSN, idxSN] += Ty[idxSN-G.Nx]
    A[idxSN, idxSN-G.Nx] -= Ty[idxSN-G.Nx]
    
    # Upper neighbors
    A[idxUN, idxUN] += Tz[idxUN]
    A[idxUN, idxUN+G.Nl] -= Tz[idxUN]
    
    # Down neighbors
    A[idxDN, idxDN] += Tz[idxDN-G.Nl]
    A[idxDN, idxDN-G.Nl] -= Tz[idxDN-G.Nl]
    
    # Apply boundary conditions (Dirichlet only)
    if bc.BL_type == 'D':
        A[idxLB, idxLB] += G.Ax[idxLB] * y[idxLB] / (G.xcmesh[0] - G.xfmesh[0])
        
    if bc.BR_type == 'D':
        A[idxRB, idxRB] += G.Ax[idxRB] * y[idxRB] / (G.xfmesh[G.Nx] - G.xcmesh[G.Nx-1])
        
    if bc.BN_type == 'D':
        A[idxNB, idxNB] += G.Ay[idxNB] * y[idxNB] / (G.yfmesh[G.Ny] - G.ycmesh[G.Ny-1])
        
    if bc.BS_type == 'D':
        A[idxSB, idxSB] += G.Ay[idxSB] * y[idxSB] / (G.ycmesh[0] - G.yfmesh[0])
        
    if bc.BU_type == 'D':
        A[idxUB, idxUB] += G.Az[idxUB] * y[idxUB] / (G.zfmesh[G.Nz] - G.zcmesh[G.Nz-1])
        
    if bc.BD_type == 'D':
        A[idxDB, idxDB] += G.Az[idxDB] * y[idxDB] / (G.zcmesh[0] - G.zfmesh[0])
        
    # Calculate assembly time
    t_nf = time.time() - t_0
    
    return A

def bc_array_FV_no_for_allB3D(G, y, bc):  
    """
    Assemble right-hand side vector for boundary conditions
    
    Parameters:
    G: grid object
    y: property values
    bc: boundary conditions
    """
    # Initialize RHS vector
    b = np.zeros(G.N)
    
    # Initialize timing
    t_0 = time.time()
    
    # Create boundary index arrays
    idx = np.arange(0, G.N)
    idxLB = np.array(np.where((idx % G.Nx) == 0))
    idxRB = np.array(np.where((idx % G.Nx) == G.Nx-1))
    idxNB = np.array(np.where(idx % G.Nl >= G.Nl-G.Nx))
    idxSB = np.array(np.where(idx % G.Nl <= G.Nx-1))
    idxUB = np.array(np.where(idx >= G.N-G.Nl))
    idxDB = np.array(np.where(idx <= G.Nl-1))
    
    # Apply boundary conditions
    # Left boundary
    Num_Left_bc = np.arange(np.shape(idxLB)[1])
    if bc.BL_type == 'D':
        b[idxLB] += bc.BL[Num_Left_bc] * G.Ax[idxLB] * y[idxLB] / (G.xcmesh[0] - G.xfmesh[0])
    else:
        b[idxLB] += bc.BL[Num_Left_bc]
        
    # Right boundary
    Num_Right_bc = np.arange(np.shape(idxRB)[1])
    if bc.BR_type == 'D':
        b[idxRB] += bc.BR[Num_Right_bc] * G.Ax[idxRB] * y[idxRB] / (G.xfmesh[G.Nx] - G.xcmesh[G.Nx-1])
    else:
        b[idxRB] += bc.BR[Num_Right_bc]
        
    # North boundary
    Num_North_bc = np.arange(np.shape(idxNB)[1])
    if bc.BN_type == 'D':
        b[idxNB] += bc.BN[Num_North_bc] * G.Ay[idxNB] * y[idxNB] / (G.yfmesh[G.Ny] - G.ycmesh[G.Ny-1])
    else:
        b[idxNB] += bc.BN[Num_North_bc]
        
    # South boundary
    Num_South_bc = np.arange(np.shape(idxSB)[1])
    if bc.BS_type == 'D':
        b[idxSB] += bc.BS[Num_South_bc] * G.Ay[idxSB] * y[idxSB] / (G.ycmesh[0] - G.yfmesh[0])
    else:
        b[idxSB] += bc.BS[Num_South_bc]
        
    # Upper boundary
    Num_Up_bc = np.arange(np.shape(idxUB)[1])
    if bc.BU_type == 'D':
        b[idxUB] += bc.BU[Num_Up_bc] * G.Az[idxUB] * y[idxUB] / (G.zfmesh[G.Nz] - G.zcmesh[G.Nz-1])
    else:
        b[idxUB] += bc.BU[Num_Up_bc]
        
    # Down boundary
    Num_Down_bc = np.arange(np.shape(idxDB)[1])
    if bc.BD_type == 'D':
        b[idxDB] += bc.BD[Num_Down_bc] * G.Az[idxDB] * y[idxDB] / (G.zcmesh[0] - G.zfmesh[0])
    else:
        b[idxDB] += bc.BD[Num_Down_bc]
        
    # Calculate assembly time
    t_nf = time.time() - t_0
    
    return b

def AI_mat_full_FV_3D(G, rock, fluid, bc):
    """
    Assemble complete finite volume system for 3D reservoir simulation
    
    Parameters:
    G: grid object
    rock: rock properties
    fluid: fluid properties
    bc: boundary conditions
    """
    # Calculate mobility
    Lambda_0 = rock.perm / fluid.mu  
    Lambda = Lambda_0
    
    # Calculate averaged mobilities
    [Lambdax_av, Lambday_av, Lambdaz_av] = lamda_av3D(Lambda, G)
    
    # Calculate transmissibilities
    [Tx, Ty, Tz] = Trans_FV3D(Lambdax_av, Lambday_av, Lambdaz_av, G)

    # Assemble system matrix and RHS
    A = A_mat_FV_no_for_allB3D(Tx, Ty, Tz, G, Lambda, bc)
    q_bc = bc_array_FV_no_for_allB3D(G, Lambda, bc) 

    return A, q_bc, Tx, Ty, Tz

# =============================================================================
# ------------------------ 3D Grid Generation Functions ---------------------
# =============================================================================

def cell_f3D(xf, yf, zf, Nx, Ny, Nz):
    """
    Generate 3D cell face coordinates
    
    Parameters:
    xf, yf, zf: face coordinates in each direction
    Nx, Ny, Nz: grid dimensions
    """
    Gx_1 = np.arange(0, Nx)
    Gy_1 = np.arange(0, Ny)
    Gz_1 = np.arange(0, Nz)
    
    meshx = np.meshgrid(Gy_1, Gz_1, xf)
    meshy = np.meshgrid(yf, Gz_1, Gx_1)
    meshz = np.meshgrid(Gy_1, zf, Gx_1)
    
    xf_vec = Tens_to_vect_new(meshx[2])
    yf_vec = Tens_to_vect_new(meshy[0])
    zf_vec = Tens_to_vect_new(meshz[1])

    return xf_vec, yf_vec, zf_vec

def cell_c3D(xf, yf, zf, Nx, Ny, Nz):
    """
    Generate 3D cell center coordinates
    
    Parameters:
    xf, yf, zf: face coordinate vectors
    Nx, Ny, Nz: grid dimensions
    """
    xc = np.zeros(Nx)
    yc = np.zeros(Ny)
    zc = np.zeros(Nz)
    Nl = Nx * Ny
    N = Nl * Nz
    
    # Calculate cell centers
    for i in range(0, Nx):
        xc[i] = xf[i] + (xf[i+1] - xf[i]) / 2
        
    hy = 0
    for i in range(0, Nl, Nx):
        yc[hy] = yf[i] + (yf[i+Nx] - yf[i]) / 2
        hy = hy + 1
        
    hz = 0
    for i in range(0, N, Nl):
        zc[hz] = zf[i] + (zf[i+Nl] - zf[i]) / 2
        hz = hz + 1
        
    mesh = np.meshgrid(yc, zc, xc)
    
    xc_vec = Tens_to_vect_new(mesh[2])
    yc_vec = Tens_to_vect_new(mesh[0])
    zc_vec = Tens_to_vect_new(mesh[1])
    
    return xc_vec, yc_vec, zc_vec

def xyz_mesh(xc_v, yc_v, zc_v, Nx, Ny, Nz, N):
    """
    Extract mesh coordinates from vectors
    
    Parameters:
    xc_v, yc_v, zc_v: center coordinate vectors
    Nx, Ny, Nz: grid dimensions
    N: total number of cells
    """
    xc_mesh = xc_v[0:Nx]
    yc_mesh = yc_v[0:Nx*Ny:Nx]
    zc_mesh = zc_v[0:N:Nx*Ny]
    
    return xc_mesh, yc_mesh, zc_mesh

# =============================================================================
# ------------------------ Grid Spacing Functions ---------------------------
# =============================================================================

def dxcenter(xc, Nx, Ny, N):
    """Calculate cell center spacing in x-direction"""
    dx = np.ones(N)
    for i in range(0, N-1):
        dx[i] = xc[i+1] - xc[i]
    return dx

def dycenter(y, Nx, Ny, N):
    """Calculate cell center spacing in y-direction"""
    dy = np.ones(N)
    for i in range(0, N-Nx):
        dy[i] = y[i+Nx] - y[i]
    return dy

def dzcenter(z, Nx, Ny, N):
    """Calculate cell center spacing in z-direction"""
    dz = np.ones(N)
    Nl = Nx * Ny
    for i in range(0, N-Nl):
        dz[i] = z[i+Nl] - z[i]
    return dz

def dxface(xf, Nx, Ny, Nz, N):
    """Calculate face spacing in x-direction"""
    dxf = np.zeros(N)
    hx = 0
    Nxf = (Nx+1) * Ny * Nz
    for i in range(0, Nxf-1):
        if i % (Nx+1) != Nx:
            dxf[hx] = xf[i+1] - xf[i]
            hx = hx + 1
    return dxf

def dyface(yf, Nx, Ny, Nz, N):
    """Calculate face spacing in y-direction"""
    dyf = np.zeros(N)
    Nl = Nx * Ny
    Nyf = Nx * (Ny+1) * Nz
    Ny2D = Nx * (Ny+1)
    hy = 0
    for i in range(0, Nyf-Nx):
        if i % (Ny2D) < Nl:
            dyf[hy] = yf[i+Nx] - yf[i]
            hy = hy + 1
    return dyf

def dzface(zf, Nx, Ny, Nz, N):
    """Calculate face spacing in z-direction"""
    dzf = np.zeros(N)
    Nl = Nx * Ny
    Nzf = Nx * Ny * (Nz+1)
    hf = 0
    for i in range(0, Nzf-Nl):
        dzf[hf] = zf[i+Nl] - zf[i]
        hf = hf + 1
    return dzf

# =============================================================================
# ------------------------ Cell Face Area Functions -------------------------
# =============================================================================

def Area_x(dyf, dzf, N):
    """Calculate cell face areas in x-direction"""
    Ax = np.zeros(N)
    for i in range(0, N):
        Ax[i] = dyf[i] * dzf[i]
    return Ax

def Area_y(dxf, dzf, N):
    """Calculate cell face areas in y-direction"""
    Ay = np.zeros(N)
    for i in range(0, N):
        Ay[i] = dxf[i] * dzf[i]
    return Ay

def Area_z(dxf, dyf, N):
    """Calculate cell face areas in z-direction"""
    Az = np.zeros(N)
    for i in range(0, N):
        Az[i] = dxf[i] * dyf[i]
    return Az

# =============================================================================
# ------------------------ Fracture Grid Generation -------------------------
# =============================================================================

def yfracture(Ny, Nfrac, Ly): 
    """
    Generate y-coordinates for fracture grid
    
    Parameters:
    Ny: number of cells in y-direction
    Nfrac: number of fractures
    Ly: domain length in y-direction
    """
    yfrac_init = Ly / Nfrac
    yfvec = np.zeros(Ny + 1)
    dy_frac = 0.01  # Fracture thickness
    
    dy_lay = (Ly / Nfrac - dy_frac) / (Ny / Nfrac - 1)
    h = 0
    
    for i in range(0, Ny + 1):
        if i % int(Ny / Nfrac) == 0:
            yfvec[i] = h * (Ly / Nfrac)
            h = h + 1
        elif i % int(Ny / Nfrac) == 1:
            yfvec[i] = yfvec[i-1] + dy_frac
    
    if Ny / Nfrac > 2:
        for i in range(0, Ny + 1):
            if i % int(Ny / Nfrac) != 0 and i % int(Ny / Nfrac) != 1:
                yfvec[i] = yfvec[i-1] + dy_lay
    
    return yfvec