#
# Author: Henrique Pereira Coutada Miranda
# Tests for the yambopy library
# Si
#
import unittest
import sys
import os
import argparse
import subprocess
import filecmp
from yambopy import *
from qepy import *
import imp

#######################################################
# Silicon GW convergence
#######################################################
sys.path.append('../tutorial/si')
import gs_si
import gw_conv_si

class TestGW_Convergence(unittest.TestCase):
    def test_ainputs(self):
        gs_si.relax()
        gs_si.scf()
        gs_si.nscf()
        gs_si.bands()
   
    def test_calcs(self):
        gs_si.run_relax()
        gs_si.run_scf()
        gs_si.run_nscf()
        gs_si.run_bands()
        gs_si.run_plot()
        gs_si.orbitals()

    def test_convergence(self):
        gw_conv_si.create_save()
        gw_conv_si.gw_convergence()
 
    def test_plot(self):
        gw_conv_si.plot_convergence()

#######################################################
# Boron Nitride Coulomb Cutoff
#######################################################
sys.path.append('../tutorial/bn')
import bse_cutoff

class TestCoulomb_Cutoff(unittest.TestCase):
    def test_calcs(self):
        bse_cutoff.run()
 
    def test_plot(self):
        bse_cutoff.plot()

#######################################################
# Parallel Bethe-Salpeter MoS2
#######################################################
sys.path.append('../tutorial/mos2')
import gs_mos2
import bse_par_mos2

class TestParallel_BSE(unittest.TestCase):
    def test_ainputs(self):
        gs_mos2.scf()
        gs_mos2.nscf()

    def test_calcs(self):
        gs_mos2.run_scf()
        gs_mos2.run_nscf()
 
    def test_parallel(self):
        bse_par_mos2.run()

    def test_plot(self):
        bse_par_mos2.plot()

def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

def clean():
        print "cleaning..."
        os.system('rm -rf relax gw bse_conv bse_cutoff bse_par bse gw_conv bands scf nscf database proj.in')
        print "done!"

if __name__ == '__main__':
    #parse options
    parser = argparse.ArgumentParser(description='Run the tutorials to test yambopy.')
    parser.add_argument('-t1', '--tutorial1', action="store_true", help='Run the GW convergence caluclation of Si')
    parser.add_argument('-t2', '--tutorial2', action="store_true", help='Run the tutorial on Coulomb-cutoff in BN')
    parser.add_argument('-t3', '--tutorial3', action="store_true", help='Run the tutorial in Parallel Bethe-Salpeter in MoS2')
    parser.add_argument('-c',  '--clean',     action="store_true", help='Clean all the data from a previous run')
    args = parser.parse_args()

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    #first test if yambo is installed
    if is_exe('yambo'):
        print "yambo not found, please install it before running the tests"
        exit()

    #first test if pw.x is installed
    if is_exe('pw.x'):
        print "pw.x not found, please install it before running the tests"
        exit()
   
    # Test for tutorial 1
    if args.tutorial1:
        clean()
        suite = unittest.TestLoader().loadTestsFromTestCase(TestGW_Convergence)
        unittest.TextTestRunner(verbosity=2).run(suite)

    # Test for tutorial 2
    if args.tutorial2:
        clean()
        suite = unittest.TestLoader().loadTestsFromTestCase(TestCoulomb_Cutoff)
        unittest.TextTestRunner(verbosity=2).run(suite)

    # Test for tutorial 3
    if args.tutorial3:
        clean()
        suite = unittest.TestLoader().loadTestsFromTestCase(TestParallel_BSE)
        unittest.TextTestRunner(verbosity=2).run(suite)

    if args.clean:
        clean()
