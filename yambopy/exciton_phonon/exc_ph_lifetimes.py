##
# Authors: FP
##

import numpy as np
from yambopy.units import ha2ev
from yambopy.tools.funcs import bose,boltzman_f
from tqdm import tqdm
from yambopy.tools.citations import citation
from joblib import Parallel, delayed

@citation("To be added")
def exc_ph_lifetimes():
    """
    This class calculates the exciton-phonon lifetimes using the following expression:

        [[1/tau_{aQ} = 2\pi/N_q \sum_{smbq} |G_{bam}(Q,q)|^2 F^s_{bm}(q,Q;T) \delta(E_{aQ}-E_{bQ+q}-s\Omega_{mq}) ]]

        - Explanation of expressions
        - Relevant citations

    Returns:
    ------------
    * Lifetimes array (Nexc_in,Nq,NTemp) with values in meV

    Parameters:
    -----------
    :: nexc_in : int
        Number of excitonic states 'a' for which to compute lifetimes.
    :: ph_temp : float
        Lattice temperature in kelvin
    :: ph_energies : float ndarray
        Phonon energies in eV [nqpts,nmodes]
    :: exc_energies : float ndarray
        Exciton energies in eV [nqpts,nexc_out]
    :: exc_ph_mat_el : cmplx ndarray
        Exciton-phonon coupling matrix elements in a.u. (hartree) at momentum Q in the full BZ [nqpts,nmodes,nexc_in,nexc_out]
    :: Q : int, optional
        Exciton Qpoint index in full BZ (python counting). Default is 0 (Gamma point).
    :: nexc_out : int, optional
        Number of excitonic final states 'b' to be summed over. Default is -1 (max number in exc_ph_mat_el).
    :: exc_temp : float, optional
        If given, effective temperature controlling excitonic population. Default is no excitonic population.
    :: broad : float, optional
        Delta function lorentzian broadening in eV (default is 5 meV).
    :: exc_energies_in : float ndarray, optional
        If given, zero-momentum states will be taken from this array. Default is exc_energies.

    Implementation steps:
        FIX UNITS!
        - Usual checks such as for PL
        - [OPT] Deg. finder for q=0 at E_in and E_out, how to treat degenerate states?
        - Integration with tetrahedron method
        - Parallelism:
            * q-point loop: parallel with joblib + tetrahedron integration: there should be an option, either broadening or tetra!
            * a-loop: serial? Typically low number
            * b-loop: parallel with joblib + tetrahedron integration
            * m-loop: parallel as well? 
        - Calculation:
            * Calculate |G|^2
            * Calculate F
            * Calculate POLE
            * Optional output: q-resolved lifetime but only if one Q
        - Pole:
            * if ph_energy < AC_thresh pole is zero; AC thresh is max(abs(acoustic energies at Gamma)*1.05 ; warning if > 5 meV

            * TO CHECK: if E_in(q=0) degenerate with E_sum(q=0) pole is zero [Hidden option]
            * TO CHECK: if E_in is exactly the same state a E_out, pole is zero [Hidden option]
    Notes (check these and do the same as lumen):
        - At q=0, degenerate states are set at exactly equal energy values (average)
        - By default the sum over exc. states does not include self-scattering 
        - By default self-scattering is excluded over full degenerate subspaces
        - By default lifetime values over degenerate states are averaged
    """
    #def get_G2_aux():
    #def get_F_aux():
    #def get_E_aux():

    # Checks
    assert exc_energies.shape[0]==ph_energies.shape[0], "q-point mismatch between excitons and phonons"
    Nqpts = ph_energies.shape[0]
    if exc_energies_in is None: exc_energies_in = exc_energies[Q]
    if np.iscomplexobj(exc_energies):    exc_energies    = exc_energies.real
    if np.iscomplexobj(exc_energies_in): exc_energies_in = exc_energies_in.real
    nexc_out_avail = min(exc_energies.shape[1],exc_ph_mat_el.shape[3])
    nexc_in_avail  = min(len(exc_energies_in),exc_ph_mat_el.shape[2])
    assert nexc_out <= nexc_out_avail, "less exciton states than requested (nexc_out)"
    assert nexc_in  <= nexc_in_avail,  "less exciton states than requested (nexc_in)"
    if nexc_out =='all': nexc_out = nexc_out_avail
    if nexc_in  =='all': nexc_in  = nexc_in_avail
    exc_energies    = exc_energies[:,:nexc_out]
    exc_energies_in = exc_energies_in[:nexc_in]
    assert ph_energies.shape[1]==exc_ph_mat_el.shape[1], "number of modes mismatch between phonon energies and matrix elements"
    
    # Scattering
    exc_ph_scatt = np.abs(exc_ph_mat_el)**2.
    
    # Delta broadening: we are using explicit lorentzian shape
    broad = broad/2.
    broad_Ha = broad/ha2ev

    # Occupation functions
    exc_min_energy = np.min(exc_energies)
    exc_occ = bose(exc_energies-exc_min_energy,exc_temp)
    ph_occ  = bose(ph_energies,ph_temp)

    # Creation of auxiliary index x=(b,m,s)
    # reshape G[q,ph,ei,eo] into |G[x,ei,q]|^2
    # construct F[x,q] 
    # construct E[x,q]
    # obtain x table (e.g. use np.unravel_index)

    # construct C=2\pi*G2*F

    # send to external function for evaluation with lorentzian
    # send to external function for evaluation with tetrahedra

#def lifetime_lorentzian(C,E,exc_energies_in,broad,...):

#def lifetime_tetra(C,E,exc_energies_in,...):


