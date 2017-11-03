# Copyright (c) 2016, Henrique Miranda
# All rights reserved.
# Authors: Alexandre Morlet, Fulvio Paleari, HM
# This file is part of yambopy
#
from netCDF4 import Dataset
import numpy as np
from itertools import product,groupby
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
        self.get_times(RT_dbs)
#
    def get_times(self,RT_dbs):
        #
        # Sorting RT dbs time-wise
        units = {'as':1e-18,'fs':1e-15,'ps':1e-12}
        s = []
        for db in RT_dbs:
            for unit in units.keys():
                if unit in db: factor = units[unit]
            s.append((float(re.findall("\d+\.\d+", db)[0])*factor,db))
        RT_dbs_ordered = sorted(s)
        self.times = [ time for time,db in RT_dbs_ordered ]
        # Reading occupations
        self.occ    = []
        self.nk_occ = []
        for i,db in enumerate(RT_dbs_ordered):
            db_occ = Dataset('%s/%s'%(self.RTsave,db[1]))
            self.occ.append(db_occ.variables['RT_carriers_delta_f'][:])
            db_occ.close()
            self.nk_occ.append(np.array( [[self.occ[i][ik*self.nbands+ib] for ib in range(self.nbands) ] for ik in range(self.nkpoints) ] ))

    def get_path2(self,path):
        """
        The function get_path in latticedb.py doesn't find the right points.

        For now, I use this workaround that is ONLY ROBUST for systems where the
        IBZ lies entirely in the positive octant of the reciprocal lattice space
        (e.g. hexagonal, whereas it might not work for fcc)
        """
        lat = self.lattice
        electrons = self.electrons
        ibz_car = np.array([ k/lat.alat for k in electrons.iku_kpoints ])
        ibz_red = car_red(ibz_car,lat.rlat)
        
        Npts = len(path)
        path_car = red_car(path,lat.rlat)
        indices = []
        # With this loop I get the INDICES of the points along the high-sym lines
        # Check isbetween in reduced coordinates
        # Check distances in cartesian coordinates
        for iK in range(Npts-1):
            start_pt  = np.array(path[iK])
            end_pt    = np.array(path[iK+1])
            start_car = np.array(path_car[iK])
            end_car   = np.array(path_car[iK+1])
            line = []
            for ik,k in enumerate(ibz_red):
                if isbetween(start_pt,end_pt,abs(k)): line.append(ik)
            if np.linalg.norm(start_car)>np.linalg.norm(end_car): line = line[::-1]
            indices.append(line)
        indices = indices[0]+indices[1]+indices[2]
        self.indices = [ ind[0] for ind in groupby(indices) ]
        self.how_many = len(self.indices)
        distance = []
        distance.append(0.)
        for ik in range(1,self.how_many):
            start_car = ibz_car[self.indices[ik-1]]
            end_car   = ibz_car[self.indices[ik]]
            distance.append(distance[ik-1]+np.linalg.norm(end_car-start_car))
        self.distance = np.array(distance)
    
    def plot_bands_and_occupations(self,path,plot='gnuplot'):
        """ Plot is 'gnuplot' or 'matplotlib' 
        """
        #Get kpath indices
        self.get_path2(path)
        #Get bands and occupations to plot
        eigs_to_plt = self.eigs[self.indices]
        occs_to_plt = [ occ[self.indices] for occ in self.nk_occ ]
        #Build multidimentsional list for each time step: [ [x],[ind],[b1 .... bN],[occ1 .... occN] ]              
        list_to_plot = []
        x_axis = np.concatenate((self.distance.reshape(self.how_many,1),np.array(self.indices).reshape(self.how_many,1)),axis=1)
        for tstep in range(self.Ndbs): 
            list_to_plot.append(np.concatenate((x_axis,eigs_to_plt,occs_to_plt[tstep]),axis=1))    
            #Print a plottable data file (i.e. gnuplot,matplotlib)
            if plot=='matplotlib':
                np.savetxt('bnds_occs_time_%s_mpl.dat'%str(tstep),list_to_plot[tstep],fmt='%4.5f',header='x\t#index\t#bands: %d cols\t#occupations: %d cols'%(self.nbands,self.nbands))
            if plot=='gnuplot':
                f = open('bnds_occs_time_%s_gnu.dat'%str(tstep),'a')
                for ib in range(self.nbands):
                    np.savetxt('bnds_occs_time_%s_gnu.dat'%str(tstep),
                                (list_to_plot[tstep][0],list_to_plot[tstep][1],list_to_plot[tstep][2+ib],list_to_plot[tstep][2+self.nbands+ib]),
                                fmt='%4.5f',header='x\t#index\t#band\t#occupations')
                f.close()

    def __str__(self):
        s =  ""
        s += "nbands pumped: %d\n"%self.nbands
        s += "from %d to %d\n"%(self.bands[0],self.bands[1])
        s += "Number of time steps: %d"%self.Ndbs
        return s         

