#
# License-Identifier: GPL
#
# Copyright (C) 2024 The Yambo Team
#
# Authors: HPC
#
# This file is part of the yambopy project
#
import numpy as np
from yambopy.units import kb

def abs2(x):
    return x.real**2 + x.imag**2
 
def lorentzian(x,x0,g):
    height=1./(np.pi*g)
    return height*(g**2)/((x-x0)**2+g**2)

def gaussian(x,x0,s,max_exp=50.,min_exp=-100.):
    height=1./(np.sqrt(2.*np.pi)*s)
    argument=-0.5*((x-x0)/s)**2
    #Avoiding undeflow errors...
    np.place(argument,argument<min_exp,min_exp)
    return height*np.exp(argument)

def boltzman_f(Eb, Bose_Temp):
    return np.exp(-Eb/(kb*Bose_Temp))

def fermi(e,max_exp=50,min_exp=-100):
    """ fermi dirac function
    """
    if e > max_exp:
        return 0
    elif e < min_exp:
        return 1
    return 1/(np.exp(e)+1)

def fermi_array(e_array,ef,invsmear):
    """
    Fermi dirac function for an array
    """
    e_array = (e_array-ef)/invsmear
    return [ fermi(e) for e in e_array]

def bose(Eb,Bose_Temp,max_exp=50,T_thr=1e-10,E_thr=0.1):
    """ 
    Bose-Einstein function (accepts ndarray)

    Eb --> in eV (should be positive)
    Bose_Temp --> in K

    If Eb=0 (e.g. phonon acoustic modes at q=0) it returns zero to avoid
    divide by zero errors: these states should be excluded from loops/calculations!
    """
    if Bose_Temp < T_thr: return np.zeros(Eb.shape) # zero temperature: no occupation
    e = Eb/(kb*Bose_Temp)
    # Ignore overflow div by zero and return zero if energy is zero 
    # (e.g. acoustic modes at q=0). 
    # THIS DOES NOT REPLACE HANDLING THE ZERO CASE EXPLICITLY IN YOUR APPLICATION
    with np.errstate(over='ignore',divide='ignore', invalid='ignore'): 
        n_be = np.where(e==0., 0.0, 1.0/(np.exp(e)-1.0))
    return n_be

def bose2(Eb, Bose_Temp, T_cut=1e-10, E_cut=0.1):
    """
    Generalised version based on the yambo `bose_f` found in
    `src/modules/mod_functions.F`

    Parameters
    ----------
    Eb : ndarray
        Array of energies in eV.
    Bose_Temp : float
        Temperature in K
    T_cut : float
        Cutoff for zero temperature (default 1e-10)
    E_cut : float
        Cutoff for the small energy approximation (default 0.1).

    Returns
    -------
    n_be : ndarray
        Bose function evaluated element-wise.
    """
    n_be = np.zeros(Eb.shape)

    # Zero temperature: no occupation (treat also negative case)
    if Bose_Temp < T_thr: 
        n_be[Eb<0.] = -1.
        return np.zeros(Eb.shape)

    kT = kb*Bose_Temp
    eps = np.finfo(np.float32).eps

    # Masks
    zero_energies  = np.abs(Eb) <= eps
    small_energies = ((np.abs(Eb)>eps) & (np.abs(Eb)<=E_cut*kT))
    large_energies = np.abs(Eb) > E_cut*kT

    # Eb -> 0 : diverging occupation
    n_BE[zero_energies] = kT / eps

    # Small |Eb|: exp(Eb/T) - 1 -> Eb/T
    n_be[small_energies] = kT / Eb[small_energies]

    # Larger |Eb|: 1/( exp(Eb/T) - 1)
    n_be[large_energies] = 1./(np.exp(Eb[large_energies]/kT)-1.)

    return n_be

