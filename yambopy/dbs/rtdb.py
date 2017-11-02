# Copyright (c) 2016, Henrique Miranda
# All rights reserved.
# Authors: Alexandre Morlet, Fulvio Paleari, HM
# This file is part of yambopy
#
from netCDF4 import Dataset
import numpy as np
from itertools import product
from yambopy.lattice import *
import os
#
class YamboRTDB():
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
        self.RTsave     = RTsave
        self.readDB()

    def readDB(self):
        #
        try: 
            db     = Dataset(self.RTfilename)
        except:
            raise IOError("Error opening database files %s in YamboOccupationsDB"%self.RTfilename)
      
        indices  = db.variables['RT_carriers_dimensions'][:]
        self.nkpoints = int(indices[0])
        self.bands    = np.array([int(indices[1]),int(indices[2])])
        self.nk_tot   = int(indices[3])
        self.nbands   = self.nk_tot/self.nkpoints
        db.close()

        #The pumped band energies only
        self.eigs   = np.array( [[self.electrons.eigenvalues_ibz[ik,band-1] for band in self.bands ] for ik in range(self.nkpoints) ] )
        
        #Get the occupations at different time steps
        RT_dbs = [ filename for filename in os.listdir(self.RTsave) if 'ndb.RT_carriers_Time' in filename ]
        self.Ndbs   = len(RT_dbs)
        if self.Ndbs == 1:
            #db_occ = Dataset(RT_dbs[0])
            db_occ = Dataset('%s/%s'%(self.RTsave,RT_dbs[0]))
            self.occ = db_occ.variables['RT_carriers_delta_f'][:]
            db_occ.close()
            #The occupations arranged along [ik,ib]
            self.nk_occ = np.array( [[self.occ[ik*self.nbands+ib] for ib in range(self.nbands) ] for ik in range(self.nkpoints) ] )
        else: self.get_times(RT_dbs)
#
    def get_times(self,RT_dbs):
        #
        # Sorting RT dbs time-wise
        units = {'as':1e-18,'fs':1e-15,'ps':1e-12}
        s = []
        for db in RT_dbs:
            for unit in units.keys():
                if unit in filename: factor = units[unit]
            s.append((float(re.findall("\d+\.\d+", db)[0])*factor,db))
        RT_dbs_ordered = sorted(s)
        self.times = [ time for time,db in RT_dbs_ordered ]
        # Reading occupations
        self.occ    = []
        self.nk_occ = []
        for db in RT_dbs_ordered:
            db_occ = Dataset('%s/%s'%(self.RTsave,db))
            self.occ.append(db_occ.variables['RT_carriers_delta_f'][:])
            db_occ.close()
            self.nk_occ.append(np.array( [[self.occ[ik*self.nbands+ib] for ib in range(self.nbands) ] for ik in range(self.nkpoints) ] ))

    def plot_bands_and_occupations(self,path):
        #
        lat = self.lattice
        lat.get_path(path)
        self.bands_kpoints = np.array( [ k_in_path_cart[:] for k_in_path_cart in lat.bands_kpoints ] )
        self.bands_indices = lat.bands_indexes
              
    def __str__(self):
        s =  ""
        s += "nbands pumped: %d\n"%self.nbands
        s += "from %d to %d\n"%(self.bands[0],self.bands[1])
        s += "Number of time steps: %d"%self.Ndbs
        return s         

