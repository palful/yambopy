##
## Authors: FP
##
import numpy as np
import os
from yambopy.dbs.excitondb import YamboExcitonDB
from yambopy.bse.exciton_matrix_elements import exciton_X_matelem
from yambopy.bse.rotate_excitonwf import rotate_exc_wf
from yambopy.kpoints import build_ktree,find_kpt
from yambopy.tools.function_profiler import func_profile
from tqdm import tqdm

def coherent_exciton_phonon_matelem(latdb,elphdb,wfdb,Qrange=[0,1],BSE_dir='bse',
                                    neigs=-1,dmat_mode='run',gbare=False,save_files=True,
                                    exph_file='COH-Ex-ph.npy',overwrite=False):
    """
    This function calculates the "coherent" first-order exciton-phonon matrix elements
    (see Eq. XX of Ref. YY). These are phonon-mediated electron-hole couplings 
    rotated in the exciton basis (similar to BSE-Hartree diagram but with phonon
    propagator instead of Coulomb interaction).

    These terms appear in the excitonic Ehrenfest dynamics (at Q=0).

    - Q is the exciton and phonon momentum

    Parameters
    ----------
    latdb : YamboLatticeDB
        The YamboLatticeDB object which contains the lattice information.
    elphdb : LetzElphElectronPhononDB
        The LetzElphElectronPhononDB object which contains the electron-phonon matrix
        elements.
    wfdb : YamboWFDB
        The YamboWFDB object which contains the wavefunction information.
    BSE_dir : str, optional
        The name of the folder which contains the BSE calculation. Default is 'bse'.
    Qrange : int list, optional
        Exciton Qpoint index range [iQ_initial, iQ_final] (python counting). Default is [0,1] (Gamma point only).
        Note that the indexing is in full BZ and not in iBZ. See wfc.kBZ to see the kpoints in full BZ
    neigs : int, optional
        Number of excitonic states included in calculation. Default is -1 (all).
    dmat_mode : str, optional
        If 'save', print dmats on .npy file for faster recalculation. If 'load', load from .npy file. Else, calculate Dmats at runtime.
    gbare : bool, optional
        if True, the bare el-ph matrix elements will be used. Default is False.
    save_files : bool, optional
        If True, the matrix elements will be saved in .npy file `exph_file`. Default is True.
    overwrite : bool, optional
        If False and `exph_file` is found, the matrix elements will be loaded from file. Default is False.
    """

    # Check if we just need to load
    if os.path.exists(exph_file) and overwrite==False:
        print(f'Loading EXCPH matrices from {exph_file}...')
        exph_mat_loaded = np.load(exph_file)
        return exph_mat_loaded

    # Load exc dbs
    exdbs = []
    for ik in range(wfdb.nkpoints):
        filename = 'ndb.BS_diago_Q%d' % (ik+1)
        excdb = YamboExcitonDB.from_db_file(latdb,filename=filename,folder=BSE_dir,\
                                            Load_WF=True, neigs=neigs)
        exdbs.append(excdb)

    # get D matrices
    Dmats = save_or_load_dmat(wfdb,mode=dmat_mode,dmat_file='Dmats.npy')

    # Calculation
    print('Calculating EXCPH matrix elements...')
    exph_mat = []
    for iQ in tqdm(range(Qrange[0],Qrange[1])):
        Q_in = wfdb.kBZ[iQ]
        latdb = wfdb.ydb
        # Determine Lkind(in)
        Ak = rotate_Akcv_Q(wfdb, exdbs, Q_in, Dmats )    
        # Get phonons
        ph_eig, elph_mat = elphdb.read_iq(iQ,convention='standard')
        # use bare matrix elements if selected
        #if gbare: 
            # call descreen directly
        elph_mat = elph_mat.transpose(1,0,2,4,3)
        # Compute ex-ph
        Akq = rotate_Akcv_Q(wfdb, exdbs, Q_in, Dmats) # Q
        tmp_exph = coherent_X_matelem(Q_in, Akq, elph_mat, wfdb.kBZ, ktree=wfdb.ktree)
        ## 0.5 for Ry to Ha
        tmp_exph = 0.5 * tmp_exph.transpose(0,1) # [nmodes, nexc ]

        exph_mat.append(tmp_exph)

    # IO
    if len(exph_mat)<2: exph_mat = exph_mat[0] # single Q-point calculation (suppress axis)
    else:               exph_mat = np.array(exph_mat) # [nQ,nmodes,nexc]
    
    if save_files: 
        if exph_file[-4:]!='.npy': exph_file = exph_file+'.npy'
        print(f'Excph coupling file saved to {exph_file}')
        np.save(exph_file,exph_mat)
    
    return exph_mat

@func_profile
def coherent_X_matelem(exe_kvec, Akq, Omn, kpts, ktree=None):
    """
    Compute g_cv rotation in the excitonic basis: <S Q| g(k-Q,Q) 

    Parameters
    ----------
    exe_kvec : array_like
        Exciton k-vector in crystal coordinates (k).
    Akq : array_like
        Wavefunction coefficients for k+Q (bra wfc) with shape (n_exe_states, 1, ns, nk, nc, nv).
    Omn : array_like
        Matrix elements of the operator O in the basis of electronic states with shape (nlambda, nk, nspin, m_bnd, n_bnd).
        ie Omn = < k+q, m, s | O(q) | n, k, s>, where m_bnd and n_bnd are final and initial bands, respectively.
        s is spin index
    kpts : array_like
        K-points used to construct the BSE with shape (nk, 3) in crystal coordinates.
    ktree : KDtree, optional
        If None, will build internally, else use the user provided
    Returns
    -------
    ex_O_mat : ndarray
        The computed exciton matrix elements with shape (nlambda, n_exe_states)
    """
    # Number of phonon modes
    nlambda = Omn.shape[0]
    #
    assert Akq.shape[1] == 1, "Works only with TDA."
    # Shape of the wavefunction coefficients
    n_exe_states, bse_calc, ns, nk, nc, nv = Akq.shape # notice last axes being (k,c,v)
    #
    # Build a k-point tree for efficient k-point searching
    if ktree is None : ktree = build_ktree(kpts)
    #
    # Find the indices of k-Q in the k-point tree
    idx_k_minus_Q = find_kpt(ktree, kpts - exe_kvec[None,:]) # k-Q needed for standard el-ph
    #
    # Extract the electron-hole channel of the Omn matrix, i.e. (k,c,v)
    Ocv = Omn[:, idx_k_minus_Q, :, nv:, :nv]
    #
    # We are now flattening following (k,c,v) order for both A and O
    Akq_conj = Akq[:,0].reshape(n_exe_states,-1).conj()
    Ocv = Ocv.reshape(nlambda,-1)
    #
    # Calculation
    ex_O_mat = np.einsum('xt,lt->lx',Akq_conj,Ocv,optimize=True)
    return ex_O_mat


def save_or_load_dmat(wfdb, mode='run', dmat_file='Dmats.npy'):
    """
    Save or load Dmats to/from .npy file `dmat_file` for faster recalculation.

     If mode=='save', print dmats on .npy file for faster recalculation. 
     If mode=='load', load from .npy file. 
     Else, calculate Dmats at runtime.
    """
    if dmat_file[-4:]!='.npy': dmat_file = dmat_file+'.npy'
    if mode=='save':
        print('Saving D matrices...')
        Dmats = wfdb.Dmat()
        np.save(dmat_file,Dmats)
        return Dmats
    elif mode=='load': 
        print('Loading D matrices...')
        if not os.path.exists(dmat_file):
            raise FileNotFoundError(f"Cannot load '{dmat_file}' - file does not exist.")
        Dmats_loaded = np.load(dmat_file)
        return Dmats_loaded
    else:
        return wfdb.Dmat()


def rotate_Akcv_Q(wfdb, exdbs, Qpt, Dmats, folder=None):
    '''
    Qpt reduced coordinates in BZ or whatever
    '''
    latdb = wfdb.ydb
    idx_BZQ = wfdb.kptBZidx(Qpt)
    iQ_isymm = latdb.symmetry_indexes[idx_BZQ]
    iQ_iBZ = latdb.kpoints_indexes[idx_BZQ]
    trev  = (iQ_isymm >= len(latdb.sym_car) / (1 + int(np.rint(latdb.time_rev))))
    symm_mat_red = latdb.lat@latdb.sym_car[iQ_isymm]@np.linalg.inv(latdb.lat)
    exe_iQIBZ = wfdb.kpts_iBZ[iQ_iBZ]
    #
    if folder is not None:
        neigs = len(exdbs[0].eigenvalues)
        filename = 'ndb.BS_diago_Q%d' % (iQ_iBZ+1)
        excdbin = YamboExcitonDB.from_db_file(latdb,filename=filename,folder=folder,\
                                              Load_WF=True, neigs=neigs)
        AQibz = excdbin.get_Akcv()
    else : AQibz = exdbs[iQ_iBZ].get_Akcv()
    #
    AQ_rot = rotate_exc_wf(AQibz,symm_mat_red,wfdb.kBZ,exe_iQIBZ,Dmats[iQ_isymm],trev,wfdb.ktree)
    
    return AQ_rot
