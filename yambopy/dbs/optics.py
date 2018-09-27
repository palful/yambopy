# Copyright (c) 2016, Henrique Miranda
# All rights reserved.
#
# This file is part of the yambopy project
# Author: Fulvio Paleari
#
from netCDF4 import Dataset
import numpy as np
from itertools import product
import collections
ha2ev  = 27.211396132
max_exp = 50
min_exp =-100.
"""function get_dipoles that can work on IP and exciton residual
   function get_occupations that can get occupation values
   functions ref_index_from_eps, absorption_coefficient_from_eps
   Functions for Abs and PL spectral functions {dimensions} with occupations attached
"""
def get_residuals(db):
    """ 
    """
    s = db.__str__()
    
