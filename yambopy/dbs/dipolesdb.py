# Copyright (c) 2017, Henrique Miranda
# All rights reserved.
#
# This file is part of the yambopy project
#
from yambopy import *
from math import sqrt
from time import time
ha2ev  = 27.211396132
max_exp = 50
min_exp =-100.

def abs2(x):
    return x.real**2 + x.imag**2
 
def lorentzian(x,x0,g):
    height=1./(np.pi*g)
    return height*(g**2)/((x-x0)**2+g**2)

def gaussian(x,x0,s):
    height=1./(np.sqrt(2.*np.pi)*s)
    argument=-0.5*((x-x0)/s)**2
    #Avoiding undeflow errors...
    np.place(argument,argument<min_exp,min_exp)
    return height*np.exp(argument)

class YamboDipolesDB():
    """
    Class to read the dipoles databases from the ``ndb.dip*`` files
    
    Can be used to for exapmle plot the imaginary part of the dielectric
    function which corresponds to the optical absorption
    """
    def __init__(self,lattice,save='SAVE',filename='ndb.dip_iR_and_P',dip_type='iR',field_dir=[1,0,0],field_dir3=[0,0,1]):
        self.lattice = lattice
        self.filename = "%s/%s"%(save,filename)
        
        #read dipoles
        try:
            database = Dataset(self.filename, 'r')
        except:
            raise IOError("Error opening %s in YamboDipolesDB"%self.filename)
            
        self.nq_ibz, self.nq_bz, self.nk_ibz, self.nk_bz = database.variables['HEAD_R_LATT'][:].astype(int)
        self.spin = database.variables['SPIN_VARS'][1].astype(int)

        # indexv is the maximum partially occupied band
        # indexc is the minimum partially empty band
        self.min_band, self.max_band, self.indexv, self.indexc = database.variables['PARS'][:4].astype(int)
        database.close()

        # determine the number of bands
        self.nbands  = self.max_band-self.min_band+1
        self.nbandsv = self.indexv-self.min_band+1
        self.nbandsc = self.max_band-self.indexc+1
        
        #read the database
        self.dipoles = self.readDB(dip_type)

        #expand the dipoles to the full brillouin zone
        self.expandDipoles(self.dipoles)

    def normalize(self,electrons):
        """ 
        Use the electrons to normalize the dipole matrix elements
        """
        eiv = electrons.eigenvalues
        nkpoints, nbands = eiv.shape
        for nk in xrange(nkpoints):
            
            eivk = eiv[nk]
            
            #create eigenvalues differences arrays
            norm = np.array([ [ec-ev for ev in eivk] for ec in eivk  ])
            
            #normalize
            for i,j in product(xrange(nbands),repeat=2):
                if norm[i,j] == 0: 
                    self.dipoles[nk,:,i,j] = 0
                else:
                    self.dipoles[nk,:,i,j] = self.dipoles[nk,:,i,j]/norm[i,j]
        dipoles = self.dipoles

    def readDB(self,dip_type):
        """
        The dipole matrix has the following indexes:
        [nkpoints, 3, nspin, nbands conduction, nbands valence]
        """
        self.dip_type = dip_type
        dipoles = np.zeros([self.nk_ibz,3,self.nbandsc,self.nbandsv],dtype=np.complex64)
        
        #check dipole db format
        filename = "%s_fragment_1"%(self.filename)
        database = Dataset(filename)
        tag1 = 'DIP_iR_k_0001_spin_0001'
        tag2 = 'DIP_iR_k_0001_xyz_0001_spin_0001'
        if tag1 in database.variables.keys():
            dipoles_format = 1
        elif tag2 in database.variables.keys():
            dipoles_format = 2
        database.close()
        
        for nk in range(self.nk_ibz):

            #open database for each k-point
            filename = "%s_fragment_%d"%(self.filename,nk+1)
            database = Dataset(filename)

            if dipoles_format == 1:
                dip = database.variables['DIP_%s_k_%04d_spin_%04d'%(dip_type,nk+1,1)][:].view(dtype=np.complex64)[:,:,:,0]
                for i in xrange(3):
                    dipoles[nk,i] = dip[:,:,i].T
            elif dipoles_format == 2:
                for i in xrange(3):
                    dip = database.variables['DIP_%s_k_%04d_xyz_%04d_spin_%04d'%(dip_type,nk+1,i+1,1)][:]
                    dipoles[nk,i] = dip[0].T+dip[1].T*1j

            #close database
            database.close()

        return dipoles
        
    def expandDipoles(self,dipoles=None,field_dir=[1,0,0],field_dir3=[0,0,1]):
        """
        Expand diples from the IBZ to the FBZ
        """
        if dipoles is None:
            dipoles = self.dipoles
            
        #check if we need to expand the dipoles to the full BZ
        lattice = self.lattice
        kpts = lattice.car_kpoints
        nks  = lattice.kpoints_indexes
        nss  = lattice.symmetry_indexes
        
        #normalize the fields
        field_dir  = np.array(field_dir)
        field_dir  = field_dir/np.linalg.norm(field_dir)
        field_dir3 = np.array(field_dir3)
        field_dir3 = field_dir3/np.linalg.norm(field_dir3)
        
        #calculate polarization directions
        field_dirx = field_dir
        field_diry = np.cross(field_dir3,field_dirx)
        field_dirz = field_dir3

        #get band indexes
        nkpoints = len(nks)
        indexv = self.min_band-1
        indexc = self.indexc-1
        nbands = self.min_band+self.nbands-1
        
        #Note that P is Hermitian and iR anti-hermitian.
        if self.dip_type == 'P':
            factor =  1.0
        else:
            factor = -1.0
           
        #save dipoles in the ibz
        self.dipoles_ibz = dipoles 
        #get dipoles in the full Brilouin zone
        self.dipoles = np.zeros([nkpoints,3,nbands,nbands],dtype=np.complex64)
        for nk_fbz,nk_ibz,ns in zip(xrange(nkpoints),nks,nss):
            
            #if time rev we conjugate
            if lattice.time_rev_list[ns]:
                dip = np.conjugate(dipoles[nk_ibz,:,:,:])
            else:
                dip = dipoles[nk_ibz,:,:,:]
            
            #get symmmetry operation
            sym = lattice.sym_car[ns].T
            #get projection operation
            pro = np.array([field_dirx,field_diry,field_dirz])
            #transformation
            tra = np.dot(pro,sym)
            
            for c,v in product(xrange(self.nbandsc),xrange(self.nbandsv)):
                #rotate dipoles
                self.dipoles[nk_fbz,:,indexc+c,indexv+v] = np.dot(tra,dip[:,c,v])
        
            #make hermitian
            for c,v in product(xrange(self.nbandsc),xrange(self.nbandsv)):
                self.dipoles[nk_fbz,:,indexv+v,indexc+c] = factor*np.conjugate(self.dipoles[nk_fbz,:,indexc+c,indexv+v])
                        
        self.field_dirx = field_dirx
        self.field_diry = field_diry
        self.field_dirz = field_dirz
        
        return dipoles, kpts
       
    def plot(self,ax,kpoint=0,dir=0,func=abs2):
        return ax.matshow(func(self.dipoles[kpoint,dir]))
        
    def ip_eps2(self,electrons,pol=1,ntot_dip=-1,GWshift=0.,broad=0.1,broadtype='l',nbnds=[-1,-1],emin=0.,emax=10.,esteps=500):
        """
        Compute independent-particle absorption (by Fulvio Paleari)

        electrons -> electrons YamboElectronsDB
        GWshift -> rigid GW shift in eV
        broad -> broadening of peaks
        broadtype -> 'l' is lorentzian, 'g' is gaussian
        nbnds -> number of [valence, conduction] bands included starting from Fermi level. Default means all are included
        emin,emax,esteps -> frequency range for the plot
        """

        #get eigenvalues and weights of electrons
        eiv = electrons.eigenvalues
        weights = electrons.weights
        nv = electrons.nbandsv
        nc = electrons.nbandsc   
 
        #get dipoles
        dipoles = self.dipoles_ibz 

        #get frequencies and im
        freq = np.linspace(emin,emax,esteps)
        eps2 = np.zeros([len(freq)])

        #Cut bands to the maximum number used for the dipoles
        if ntot_dip>0: 
            eiv = eiv[:,:ntot_dip]
            nc=ntot_dip-nv

        #Print band gap values and apply GW_shift
        eiv = electrons.energy_gaps(GWshift)

        #Check bands to include in the calculation
        if nbnds[0]<0: nbnds[0]=nv
        if nbnds[1]<0: nbnds[1]=nc
        iv = nv-nbnds[0] #first valence
        lc = nv+nbnds[1] #last conduction

        #choose broadening
        if "l" in broadtype:
            broadening = lorentzian
        else:
            broadening = gaussian

        pols = np.array(pols)
        na = np.newaxis
        #calculate epsilon
        for c,v in product(range(nv,lc),range(iv,nv)):
            #get electron-hole energy and dipoles
            ecv  = eiv[:,c]-eiv[:,v]
            dip2 = np.sum(abs2(dipoles[:,pols,c-nv,v]),axis=1)

            #make dimensions match
            dip2a = dip2[na,:]
            ecva  = ecv[na,:]
            freqa = freq[:,na]
            wa    = weights[na,:]       
  
            #calculate the lorentzians 
            broadw = broadening(freqa,ecva,broad)
   
            #scale broadening with dipoles and weights
            epsk =  wa*dip2a*broadw

            #integrate over kpoints
            eps2 += np.sum(epsk,axis=1)

        return freq, eps2

    def get_eps(self,electrons,efield=[1.,0.,0.],nbands=[-1,-1],emin=0.,emax=10.,esteps=100,broad=0.1):
        """
        Compute the macroscopic complex independent-particle dielectric function (by Fulvio Paleari)
        We use the IP responce function X_0
        So we have eps(w) = 1+lim_{q->0}v(q)X_0(w,q)
        Yambo test --> yambo -o c and Chimod="IP"

        electrons -> electrons YamboElectronsDB
        efield -> electric field direction
        broad -> imaginary part of the energies (in eV)
        nbnds -> number of [valence, conduction] bands included starting from Fermi level. Default means all are included
        emin,emax,esteps -> frequency range for the plot (in eV)
        """
        
        # First set up constants and dimensions
        q0_norm = 1e-05
        e0_norm = q0_norm # This ensures cancellation in the final formula
        efield  = np.array(efield)
        efield  = efield/np.linalg.norm(efield)*e0_norm
        
        s=2 # For now we assume a spin-unpolarised material
        D_vol = self.lattice.D_vol_au # We need the unit cell volume

        # Quantities that will be needed
        dipoles = self.dipoles
        eiv     = electrons.eigenvalues
        weights = electrons.weights
        nv      = electrons.nbandsv
        nc      = electrons.nbandsc
        nk      = self.nk_bz
        
        #Check bands to include in the calculation
        if nbands[0]<0: nbnds[0]=nv
        if nbands[1]<0: nbnds[1]=nc
        iv = nv-nbands[0] #first valence
        lc = nv+nbands[1] #last conduction        

        #Compute X_0
        w = np.linspace(emin,emax,esteps)
        chi_0 = np.zeros([len(w)], dtype=np.complex128)
        for k in xrange(nk):
            eivk = eiv[k]
            for c,v in product(xrange(nv,lc),xrange(iv,nv)):
                e_trans = eivk[c]-eivk[v]
                G1 = -1./(w-e_trans+I*broad) #This is the resonant term
                G2 =  1./(w+e_trans+I*broad) #This is the anti-resonant term
                dip2 = 0.
                for i in range(3): dip2+=abs2(efield[i]*dipoles[k,i,c,v])
                dip2 = dip2*weights[k]
                chi_0+=dip2*(G1+G2)
        #Dielectric function with the correct dimensions
        eps = 1. + ha2ev/D_vol*s*4.*np.pi/q0_norm**2.*chi_0
        return w, eps

    def __str__(self):
        s = ""
        s += "\nkpoints:\n"
        s += "nk_ibz : %d\n"%self.nk_ibz
        if self.expand: s += "nk_bz  : %d\n"%self.nk_bz
        s += "\nnumber of bands:\n"
        s += "nbands : %d\n" % self.nbands
        s += "nbandsv: %d\n" % self.nbandsv
        s += "nbandsc: %d\n" % self.nbandsc
        s += "indexv : %d\n" % (self.min_band-1)
        s += "indexc : %d\n" % (self.indexc-1)
        if self.expand:
            s += "field_dirx: %10.6lf %10.6lf %10.6lf\n"%tuple(self.field_dirx)
            s += "field_diry: %10.6lf %10.6lf %10.6lf\n"%tuple(self.field_diry)
            s += "field_dirz: %10.6lf %10.6lf %10.6lf\n"%tuple(self.field_dirz)
        return s

if __name__ == "__main__":
    ddb = DipolesDB()
    ddb.get_databases()
    print ddb
