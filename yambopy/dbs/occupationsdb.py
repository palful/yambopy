# 1st version by Fulvio Paleari
# This file is part of yambopy
#
from netCDF4 import Dataset
import numpy as np
from itertools import product
#
class YamboOccupationsDB():
    """
    Class to read information about occupations in the case of a carrier "pumping" by yambo

    Arguments:
        ``lattice``: instance of YamboLatticeDB or YamboSaveDB
        ``electrons``: instance of YamboElectronsDB

        ``save`` and ``filename``: where to read for lattice and electrons
        ``RTsave`` and ``RTfilename``: where to read occupations
    """
    def __init__(self,lattice,electrons,save='SAVE',filename='ns.db1',RTsave='SAVE',RTfilename='ndb.RT_carriers'):
        self.lattice    = lattice
        self.electrons  = electrons
        self.filename   = '%s/%s'%(save,filename)
        self.RTfilename = '%s/%s'%(RTsave,RTfilename)
        self.readDB()
        self.get_trimmed_data()

    def readDB(self):
        #NB: now reads only occupations at setup (Time=0.0).
        #    To do in the future:
        #    - read occupations at every time step that is found
        #    - prepare animations of carrier relaxation
        try: 
            db     = Dataset(self.RTfilename)
            db_occ = Dataset('%s_Time___0.0_as___'%self.RTfilename)
        except:
            raise IOError("Error opening database files %s in YamboOccupationsDB"%self.RTfilename)
        
        indices  = db.variables['RT_carriers_dimensions'][:]
        self.nkpoints = int(indices[0])
        self.bands    = np.array([int(indices[1]),int(indices[2])])
        self.nk_tot   = int(indices[3])
        self.nbands   = self.nk_tot/self.nkpoints
        self.occ = db_occ.variables['RT_carriers_delta_f'][:]
        db.close()
        db_occ.close()

    def get_trimmed_data(self):
        #The pumped band energies and occupations are arranged along [ik,ib]
        self.eigs   = np.array( [[self.electrons.eigenvalues_ibz[ik,band-1] for band in self.bands ] for ik in range(self.nkpoints) ] )
        self.nk_occ = np.array( [[self.occ[ik*self.nbands+ib] for ib in range(self.nbands) ] for ik in range(self.nkpoints) ] )

    def __str__(self):
        s =  ""
        s += "nbands pumped: %d\n"%self.nbands
        s += "from %d to %d"%(self.bands[0],self.bands[1])
        return s         

