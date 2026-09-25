##
# Authors: FP
##

import numpy as np
from yambopy.units import ha2ev
from yambopy.tools.funcs import bose,boltzman_f
from yambopy.kpoints import build_ktree, find_kpt
from yambopy.tools.tetra import *
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
    :: lat : YamboLatticeDB object
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
    :: njobs : int, optional
        Number of jobs for joblib parallelization (Default 1).
    :: free_memory : bool, optional
        If True, progressively delete input array when auxiliary quantities are constructed (default False)

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
    def get_G2_aux(G,PH_E,PH_thresh):
        """
        - reshape G[q,ph,ei,eo] into |G[ei,q,x]|^2
        - obtain x=(ph,eo) table
        - apply checks on degeneracies and zero energies
        """
        # Set to zero the scattering if phonon energy is ~0
        # acoustic phonons are already excluded as G set to czero
        #G[PH_E<PH_thresh]=0.
        # Degenerate states handling
        # Reshape
        nq,nm,ne_i,ne_o = G.shape
        exc_ph_aux = G.transpose(2,0,1,3).reshape(nq,ne_i,nm*ne_o)
        exc_ph_aux = np.abs(exc_ph_aux)**2.
        table = np.array(np.unravel_index(np.arange(nm*ne_o),(nm,ne_o))).T
        return exc_ph_aux, table

    def get_F_aux(ph_occ,exc_occ):
        """
        - get occupation functions F[q,x]
        """
        # Occupation functions [q,ph,eo]
        nq,nm = ph_occ.shape
        ne_o  = exc_occ.shape[1]
        # EXC_OCC must be evaluated in Q+q
        F = np.empty((2,nq,nm,ne_o))
        F[0] = ph_occ[:,:,None] + exc_occ[:,None,:] + 1. # ph. em.
        F[1] = ph_occ[:,:,None] - exc_occ[:,None,:]      # ph. abs.
        # Reshaped occupation functions [q,x]
        F = F.reshape(2,nq,nm*ne_o)
        return F

    def get_E_aux(ph_e,exc_e):
        """
        - get pole energy E[q,x]
        """
        # Pole energy [q,ph,eo]
        nq,nm = ph_energies.shape
        ne_o  = exc_energies.shape[1]
        E = np.empty((2,nq,nm,ne_o))
        E[0] = exc_energies[:,None,:]+ph_energies[:,:,None] # ph. em.
        E[1] = exc_energies[:,None,:]+ph_energies[:,:,None] # ph. abs.
        # Reshaped pole energy [q,x]
        E = E.reshape(2,nq,nm*ne_o)
        return E

    # Checks
    assert exc_energies.shape[0]==ph_energies.shape[0], "q-point mismatch between excitons and phonons"
    nqpts = ph_energies.shape[0]
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
    assert len(exc_ph_mat_el.shape)==4, "matrix elements G[Q_in] must have dimensions [nqpts,nmodes,nexc_in,nexc_out]" 
    assert ph_energies.shape[1]==exc_ph_mat_el.shape[1], "number of modes mismatch between phonon energies and matrix elements"

    # Threshold for zero energy phonons (1.05*MAX(q=0 acoustic energies))
    PH_thresh = np.max(np.abs(ph_energies[0,:3]))*1.05
    if PH_thresh>0.05: print("[WARNING] High threshold for zero phonon energies (>5 meV), check phonon dispersion")
    
    # Evaluate exc_out energies at q+Q
    # I NEED elph.qpoints
    # THIS PART CAN BE SENT TO exc_ph_get_inputs with mode=='life'
    # Modify the docstring to emphasize this
    # ALSO IT WOULD BE NICE TO REMOVE THE DEPENDENCY ON YamboLatticeDB ALSO
    # FOR TETRA CASE...
    qpts = lat.red_kpoints
    if ktree is None : ktree = build_ktree(qpts)
    idx_Q_plus_q = find_kpt(ktree, qpts + qpts[Q,:])  # q+Q
    exc_energies = exc_energies[idx_Q_plus_q,:]

    # Creation of auxiliary index x=(b,m,s)
    
    # Construct scattering strengths G2[q,x] and apply thresholds
    exc_ph_aux, x_table = get_G2_aux(exc_ph_mat_el,ph_energies,PH_thresh)
    if free_memory: del exc_ph_mat_el
    
    # construct occupation functions F[q,x] 
    exc_min_energy = np.min(exc_energies)
    exc_occ = bose(exc_energies-exc_min_energy,exc_temp)
    ph_occ  = bose(ph_energies,ph_temp)
    F_aux = get_F_aux(ph_occ,exc_occ) # move three lines above
    if free_memory: del ph_occ, exc_occ
    
    # construct pole energys E[q,x]
    E_aux = get_E_aux(ph_energies,exc_energies)
    if free_memory: del ph_energies, exc_energies
    
    # construct generalized scattering strength C=2\pi*G2*F
    # and poles E_o(Q+q)+-E_ph(q)
    N_aux = nmodes*nexc_out*2
    C = np.zeros(N_aux)
    C[:nmodes*nexc_out] = exc_ph_aux*F_aux[0]
    C[nmodes*nexc_out:] = exc_ph_aux*F_aux[1]
    if free_memory: del exc_ph_aux,F_aux
    E = np.zeros(N_aux)
    E[:nmodes*nexc_out] = E_aux[0]
    E[nmodes*nexc_out:] = E_aux[1]
    if free_memory: del E_aux

    # send to external function for evaluation with lorentzian
    # parallelised here with joblib
    broad = broad/2. # We are using explicit Lorentzian shape
    invtau = np.array( list( tqdm( Parallel(return_as="generator",n_jobs=njobs)(delayed(lifetime_lorentzian)(exc_energies[iE_in]/ha2ev,C,E/ha2ev,broad/ha2ev) for iE_in in range(nexc_in)), total=nexc_in, desc="Exc-ph lifetime calculation")))
    # serial check
    #invtau = np.zeros(nexc_in)
    #for iE_in tqdm(range(nexc_in),desc="Exc-ph lifetime calculation") :
    #    E_in = exc_energies_in[iE_in]
    #    invtau[iE_in] = lifetime_lorentzian(C,E/ha2ev,E_in/ha2ev,broad/ha2ev)
        
    # send to external function for evaluation with tetrahedra
    # use internal tetrahedron parallelization
    RLAT = lat.rlat*lat.alat[0]
    invtau = lifetime_tetra(C,E/ha2ev,exc_energies_in/ha2ev,qpts,RLAT,njobs=njobs)

    return invtau

def lifetime_lorentzian(C,E,E_in,eta):
    """
    Evaluation with delta broadening

    NB: All quantities in hartree
    """
    # Energy conservation
    delta_funct = 1./((E_in-E)**2.+ eta**2.)
    # Dimensions
    delta_funct = delta_funct * eta/np.pi/nqpts
    # Sum over q and x
    invtau = np.einsum('qx,qx->',C,delta_funct,optimize=True)
    return invtau * ha2ev * 1000. # return value in meV
    
def lifetime_tetra(C,E,E_in,qpts,RLAT,njobs=1):
    """
    Evaluation with optimized tetrahedron method

    NB: all quantities in hartree
    """
    tetra = get_tetrahedra_mesh(nk1,nk2,nk3,qpts,RLAT)
    invtau = spectra_tetrahedron(E,tetra,E_in,nspin=1,matels=C,njobs=njobs)
    return invtau * ha2ev * 1000.

