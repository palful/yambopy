#!/usr/bin/env python2.7 
#############################################################################################
# Unfolding of the electronic structure.
# Program adapted for reading values of Quantum Espresso.
#
# Authors: Alejandro Molina-Sanchez and Henrique Miranda
# Version of 24 of February of 2014.
#############################################################################################

import re
import os, sys
from time import time
import subprocess
from subprocess import PIPE
from math import sqrt, degrees, acos, cos, sin, atan, pi
import xml.etree.ElementTree as ET
import numpy as np
from numpy import array, dot, cross, complex, zeros
from numpy.linalg import solve, norm
from sys import stdout
from netCDF4 import Dataset, MFDataset

def load(x,n):
    bar_length = 60
    x+=1
    ratio = x/float(n)
    c = int(ratio * bar_length)
    stdout.write("["+"="*c+" "*(bar_length-c)+"] %03.3f%%" % (ratio*100))
    if (x==n): stdout.write("\n")
    stdout.flush()
    stdout.write("\r")

# Parameters of the calculations
#mos2
#apc   = 6.02625
#asc   = 6.02625*3
#grir
apc   = 4.5909880105
asc   = 4.5909880105*3
size  = asc/apc
cpc   = 30
csc   = 30
angle =  0.0 

nb_sc =  170 
nb_pc =  10 

sc_folder = '/work/users/hmiranda/mos2/qe/SV/5x5/band/'
pc_folder = '/work/users/hmiranda/mos2/qe/bulk/PW/band/'

# Directories holding the eigenvectors and eigenvalues
if (len(sys.argv)>2):
    sc_folder = sys.argv[1]
    pc_folder = sys.argv[2]

print 'Primitive lattice constant   ', apc
print 'Supercell lattice constant   ', asc
print 'Angle                        ', angle

rot = array([[  cos(angle) , sin(angle), 0.0 ],
             [ -sin(angle) , cos(angle), 0.0 ],
             [     0.0     ,    0.0    , 1.0 ]])

#unit cell vectors

a1 = array([ 1.0,       0.0, 0.0 ])
a2 = array([-0.5, sqrt(3)/2, 0.0 ])
a3 = array([ 0.0,       0.0, 1.0 ])

# Grid of q-vectors
grid = array([ 50, 25, 50 ])
kg = array([[   0.0,   0.0, 0.0 ],
            [   0.5,   0.0, 0.0 ],
            [ 1./3., 1./3., 0.0 ],
            [   0.0,   0.0, 0.0 ]])

# Calculation of the Brillouin zone
v  = dot(a1,cross(a2,a3))
b1 = cross(a2,a3)/v
b2 = cross(a3,a1)/v
b3 = cross(a1,a2)/v

lat_pc1 = apc*a1
lat_pc2 = apc*a2
lat_pc3 = cpc*a3

lat_sc1 = asc*a1
lat_sc2 = asc*a2
lat_sc3 = csc*a3

b_pc1 = b1/apc
b_pc2 = b2/apc
b_pc3 = b3/cpc
              
b_sc1 = b1/asc
b_sc2 = b2/asc
b_sc3 = b3/csc

# Generation of the q-path for primitive and supercell
# We write in lattice coordinates for QE 
def paths(gri,kw,size,c1,c2,c3,rot):
    k_red = [] ; k_lat = [] ; k_sc = []
    for i in xrange(len(gri)):
        for j in xrange(gri[i]):
            k_red.append(kw[i] + float(j)/gri[i]*(kw[i+1]-kw[i]))
    f1 = open('list-pc.dat','w')
    f2 = open('list-sc.dat','w')
    f1.write( str(len(k_red)) + "\n" )
    f2.write( str(len(k_red)) + "\n" )
    for i in xrange(len(k_red)):
        k_lat.append( k_red[i][0]*c1 + k_red[i][1]*c2 + k_red[i][2]*c3  )
        k_sc.append( size*dot(k_lat[i],rot) )
        #convert reciprocal cartesiaan coordinates to reduced reciprocal coordinates
        k_rlat = solve(array([c1,c2,c3]).T, k_sc[i].T)
        f1.write( "%12.5e  %12.5e  %12.5e   %8.5lf \n" %  (k_red[i][0],k_red[i][1],k_red[i][2],1.0) )
        f2.write( "%12.5e  %12.5e  %12.5e   %8.5lf \n" %  (k_rlat[0]  ,k_rlat[1]  ,k_rlat[2]  ,1.0) )
    f1.close()
    f2.close()
    return k_red, k_lat, k_sc

# Call function to generate the q-path 
print 'Generation of the q-path'
q_red, q_lat, q_sc = paths(grid,kg,size,b1,b2,b3,rot)
qmesh = len(q_red)
print 'Generated list-pc and list-sc\nNumber of k-points', qmesh

gsc = open('bands-sc.dat','w')
gpc = open('bands-pc.dat','w')


##############################################################################
#   Reading primitive cell
##############################################################################
print "\n\nReading primitive cell:" 
print pc_folder+'out.nc'
pcfile     = Dataset(pc_folder+'out.nc', 'r', format='NETCDF4')
nbands_pc  = len(pcfile.dimensions['num_bands'])
nkpoints_pc= len(pcfile.dimensions['num_kpoints'])
eig_pc     = pcfile.variables['eigenvalues'][:]
rprim_pc   = pcfile.variables['lattice_vectors'][:]
eivec_pc   = pcfile.variables['eigenvecs'][:]
kpoints_pc = pcfile.variables['kpoints'][:]
gpoints_pc = pcfile.variables['gvectors'][:]
ng_pc      = pcfile.variables['num_gvectors'][:].astype(int)

print "rprim:\n",rprim_pc
print "nbands:", nbands_pc
print "kpoints:", kpoints_pc.shape
print "eival_pc:", eig_pc.shape
print "gpoints:", gpoints_pc.shape
print "kpoints:", kpoints_pc.shape
print "ng:", ng_pc.shape
pcfile.close()

##############################################################################
#   Reading supercell
##############################################################################
print "\n\nReading supercell cell:",sc_folder
print sc_folder+'out.nc'
scfile     = Dataset(sc_folder+ 'out.nc', 'r', format='NETCDF4')
nbands_sc  = len(scfile.dimensions['num_bands'])
nkpoints_sc= len(scfile.dimensions['num_kpoints'])
eig_sc     = scfile.variables['eigenvalues'][:]
evec_sc    = scfile.variables['eigenvecs']
rprim_sc   = scfile.variables['lattice_vectors'][:]
kpoints_sc = scfile.variables['kpoints'][:]
gpoints_sc = scfile.variables['gvectors'][:]
ng_sc      = scfile.variables['num_gvectors'][:]

print "rprim:\n",rprim_sc
print "nbands:", nbands_sc
print "kpoints:", kpoints_sc.shape
print "eival_sc:", eig_sc.shape
print "gpoints:", gpoints_sc.shape
print "kpoints:", kpoints_sc.shape
print "ng:", ng_sc.shape

#make some safety checks
if (asc != rprim_sc[0][0]):
    print "specified asc %lf is different from the read lattice %lf" %(asc,rprim_sc[0][0])
    print "which one to use? (1,2)"
    asc = (asc,rprim_sc[0][0])[int(raw_input())-1]

if (apc != rprim_pc[0][0]):
    print "specified apc %lf is different from the read lattice %lf" %(apc,rprim_pc[0][0])
    print "which one to use? (1,2)"
    apc = (apc,rprim_pc[0][0])[int(raw_input())-1]

if ( nbands_pc != nb_pc ):
    print "specified nbands_pc %d is different from the bands %d" %(nb_pc,nbands_pc)
    print "which one to use? (1,2)"
    nb_pc = (nb_pc,nbands_pc)[int(raw_input())-1]

if ( nbands_sc != nb_sc ):
    print "specified nbands_sc %d is different from the bands %d" %(nb_sc,nbands_sc)
    print "which one to use? (1,2)"
    nb_sc = (nb_sc,nbands_sc)[int(raw_input())-1]

w_sc   = zeros([qmesh,nb_sc])
##########################################################
#   Start program
##########################################################

print "\n\nstart the projection..."
for iq in xrange(qmesh):
    print "Treating kpoint", iq + 1 
    print "Reading Wavevectors"
    evc = array(evec_sc[iq,:,:,0]+evec_sc[iq,:,:,1]*complex(0,1))

    g_sc = dict()
    print 'Reading Supercell G-vectors'
    for ig in xrange(ng_sc[iq]):
        load(ig,ng_sc[iq])
        x,y,z = gpoints_sc[iq][ig]
        w = x*b_sc1 + y*b_sc2 + z*b_sc3 #scaling
        w = dot(rot,w) #rotations
        w = "%8.4lf %8.4lf %8.4lf" % (w[0],w[1],w[2]) #truncation
        g_sc[w] = ig #create dictionary
    
    g_contain = [0]*ng_pc[iq]
    for ig in xrange(ng_pc[iq]):
        load(ig,ng_pc[iq])
        x,y,z = gpoints_pc[iq][ig]
        w = x*b_pc1 + y*b_pc2 + z*b_pc3 #scaling
        w = "%8.4lf %8.4lf %8.4lf" % (w[0],w[1],w[2]) #truncation
        try:
            g_contain[ig] = g_sc[w]
        except KeyError:
            print "Missing g-point %4d"% ig, w 
            g_contain[ig] = 0

    # Projection
    time1 = time()
    print "Projection"
    for ib in xrange(nb_sc): 
        load(ib,nb_sc)
        x = 0.0
        for ig in xrange(ng_pc[iq]):
            x += evc[ib][g_contain[ig]]*(evc[ib][g_contain[ig]].conjugate())
        w_sc[iq][ib] = abs(x)
    print "took: %.2lf s" % (time()-time1)
scfile.close()
print 'End of projection calculations...'
print 'Writing bands and projection in files...'

for ib in xrange(nb_sc):
  for iq in xrange(qmesh):
    gsc.write( "%5d %10.5e %10.5le\n" %  (iq,eig_sc[iq][ib],w_sc[iq][ib]) )
  gsc.write("\n")
gsc.close()
  
for ib in xrange(nb_pc):
  for iq in xrange(qmesh):
    gpc.write( "%5d %10.5e\n" % (iq,eig_pc[iq][ib]) )
  gpc.write("\n")
gpc.close()

print 'Done!'

