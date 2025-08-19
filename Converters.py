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
import warnings

# =============================================================================
# ---------------------------- Length Conversions ----------------------------
# =============================================================================

def feet2meter(ft):
    """Convert feet to meters"""
    return 0.3048 * ft

def cm2meter(cm):
    """Convert centimeters to meters"""
    return 0.01 * cm

def meter2feet(mts):
    """Convert meters to feet"""
    return 3.28084 * mts

def meter2cm(mts):
    """Convert meters to centimeters"""
    return 100 * mts

def Length_converter(L, units):
    """
    Convert length to meters (SI unit)
    
    Parameters:
    L: length value
    units: input units ('ft', 'cm', 'm')
    
    Returns:
    L: length in meters
    """
    if units == 'ft':
        L = feet2meter(L)
    elif units == 'cm':
        L = cm2meter(L)
    elif units == 'm':
        pass  # Already in SI units
    else:      
        warnings.warn(f'Length units "{units}" are not supported. Supported units: ft, cm, m')
    return L

# =============================================================================
# --------------------------- Pressure Conversions ---------------------------
# =============================================================================

def psi2Pa(psi):
    """Convert psi to Pascals"""
    return 6894.76 * psi

def Pa2psi(Pa):
    """Convert Pascals to psi"""
    return 0.000145038 * Pa

def atm2Pa(atm):
    """Convert atmospheres to Pascals"""
    return 101325 * atm

def Pa2atm(Pa):
    """Convert Pascals to atmospheres"""
    return Pa * 9.8692e-6

def Pressure_converter(P, units):
    """
    Convert pressure to Pascals (SI unit)
    
    Parameters:
    P: pressure value
    units: input units ('psi', 'atm', 'Pa')
    
    Returns:
    P: pressure in Pascals
    """
    if units == 'psi':
        P = psi2Pa(P)
    elif units == 'atm': 
        P = atm2Pa(P)
    elif units == 'Pa': 
        pass  # Already in SI units
    else:   
        warnings.warn(f'Pressure units "{units}" are not supported. Supported units: psi, atm, Pa')
    return P

# =============================================================================
# ---------------------------- Density Conversions ---------------------------
# =============================================================================

def lbft2kgm(lbft3):
    """Convert lb/ft³ to kg/m³"""
    return (3.28084**3) / 2.20462 * lbft3

def kgm2lbft(kgm3):
    """Convert kg/m³ to lb/ft³"""
    return (2.20462) / (3.28084**3) * kgm3

def Density_converter(rho, units):
    """
    Convert density to kg/m³ (SI unit)
    
    Parameters:
    rho: density value
    units: input units ('lbft3', 'kgm3')
    
    Returns:
    rho: density in kg/m³
    """
    if units == 'lbft3':
        rho = lbft2kgm(rho)
    elif units == 'kgm3': 
        pass  # Already in SI units
    else:   
        warnings.warn(f'Density units "{units}" are not supported. Supported units: lbft3, kgm3')
    return rho

# =============================================================================
# ------------------------- Permeability Conversions ------------------------
# =============================================================================

def Da2m2(Da):
    """Convert Darcy to m²"""
    return Da * 9.869233e-13

def m22Da(m2):
    """Convert m² to Darcy"""
    return m2 * 1.013249965828145e12

def Permeability_converter(K, units):
    """
    Convert permeability to m² (SI unit)
    
    Parameters:
    K: permeability value
    units: input units ('Da', 'm2')
    
    Returns:
    K: permeability in m²
    """
    if units == 'Da':
        K = Da2m2(K)
    elif units == 'm2': 
        pass  # Already in SI units
    else:   
        warnings.warn(f'Permeability units "{units}" are not supported. Supported units: Da, m2')
    return K

# =============================================================================
# -------------------------- Viscosity Conversions --------------------------
# =============================================================================

def cp2Pas(cp):
    """Convert centipoise to Pascal-seconds"""
    return 0.001 * cp

def Pas2cp(Pas):
    """Convert Pascal-seconds to centipoise"""
    return 1000 * Pas

def Viscosity_converter(mu, units):
    """
    Convert viscosity to Pa·s (SI unit)
    
    Parameters:
    mu: viscosity value
    units: input units ('cp', 'Pas')
    
    Returns:
    mu: viscosity in Pa·s
    """
    if units == 'cp':
        mu = cp2Pas(mu)
    elif units == 'Pas': 
        pass  # Already in SI units
    else:   
        warnings.warn(f'Viscosity units "{units}" are not supported. Supported units: cp, Pas')
    return mu

# =============================================================================
# --------------------------- Flow Rate Conversions -------------------------
# =============================================================================

def stbday2m3s(stbday):
    """Convert stock tank barrels per day to m³/s"""
    return 1.84013e-6 * stbday  # More accurate conversion factor

def m3s2stbday(m3s):
    """Convert m³/s to stock tank barrels per day"""
    return m3s / 1.84013e-6

def Flux_converter(q, units):
    """
    Convert flow rate to m³/s (SI unit)
    
    Parameters:
    q: flow rate value
    units: input units ('stbday', 'm3s')
    
    Returns:
    q: flow rate in m³/s
    """
    if units == 'stbday':
        q = stbday2m3s(q)
    elif units == 'm3s': 
        pass  # Already in SI units
    else:      
        warnings.warn(f'Flow rate units "{units}" are not supported. Supported units: stbday, m3s')
    return q