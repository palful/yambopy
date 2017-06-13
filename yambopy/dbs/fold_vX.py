#
# Temporary class to fold static screening
# (i.e. em1s / vX / chi) in supercells. 
# Maybe it can be merged with em1s.py. 
#
from yambopy import *
from itertools import product

class fold_vX():
    """Starting points: fully computed vX in unit cell, trash run of vX in supercell
       with consistent {qG} structure (k-point mesh, blocks cutoff).
    """
    def __init__(self,uc_lattice_path,uc_screening_path,sc_lattice_path,sc_screening_path,out_path='./em1s_folded'):
        # (0) read dbs
        ylat  = YamboLatticeDB(save=uc_lattice_path)
        ychi  = YamboStaticScreeningDB(save=uc_screening_path)
        yslat = YamboLatticeDB(save=sc_lattice_path)
        yschi = YamboStaticScreeningDB(save=sc_screening_path)
        self.ychi  = ychi
        self.yschi = yschi
        # now I have Gvectors, sgvectors, iQpts, isqpts in cartesian coordinates
        self.gvectors  = ychi.gvectors
        self.sgvectors = yschi.gvectors
        self.iqpts  = ychi.qpoints
        self.isqpts = yschi.qpoints
        # (1) expand uc and sc in full BZ
        ylat.expandKpoints(qpoints=self.iqpts)
        self.ylat = ylat
        self.qpts = ylat.car_kpoints 
        yslat.expandKpoints(qpoints=self.isqpts)
        self.yslat = yslat
        self.sqpts = yslat.car_kpoints
        # keep track of the positions of the ibz q-points in the expanded list
        uc_map = point_matching(self.qpts,self.iqpts,double_check=False)
        sc_map = point_matching(self.sqpts,self.isqpts,double_check=False)
        self.iqpts_ind  = ylat.kpoints_indexes
        self.isqpts_ind = yslat.kpoints_indexes
        #
        self.GG   = len(self.gvectors)
        self.sGG  = len(self.sgvectors)
        self.qq   = len(self.qpts)
        self.iqq  = len(self.iqpts)
        self.sqq  = len(self.sqpts)
        self.isqq = len(self.isqpts)
        """ Each Q in uc is connected to a q in sc by a sc lattice vector g_Q
            The set of g_Qs allow us to pass from {QG} to {qg} and then from
            {Q+G1,Q+G2} to {q+g1,q+g2}: we have Q=q+g_Q, g1=g_Q+G1, g2=g_Q+G2
        """
        # (4) Qqg contains the following indices: (Q,q,g) in {qpts,sqpts,sgvectors}
        self.Qqg = self.get_Qqg()
        #
        self.new_x = self.match_X()
        #
        self.yschi.saveDBS(out_path,new_X=self.new_x)
        print('folded DBS saved')
        #
        # (2) function to recover the equivalent IBZ kpoint from a full-BZ kpoint
    def make_irr(self,map_,ind_Q,Q,iqpts_list):
        ind_iQ = map_[ind_Q]
        iQ =iqpts_list[ind_iQ]
        return ind_iQ, iQ
    # (3) Get a list of the g_Q lattice vectors
    def get_Qqg(self):
        qtemp = np.array( [[Q-q for q in self.sqpts] for Q in self.qpts] )
        thr = 1e-5
        Qqg_list = []
        for i,Q in enumerate(qtemp):
            for j,q in enumerate(Q):
                for k,g in enumerate(self.sgvectors):
                    if (abs(q-g)<thr).all(): Qqg_list.append([i,j,k])
        return np.array(Qqg_list)
    # (5) Assign Xuc_G1G2(f(Q)) --> Xsc_g1g2(f(q))
    def match_X(self):
        new_x = np.zeros([self.sqq,self.sGG,self.sGG],dtype=np.complex64)
        v_scale = self.ychi.volume/self.yschi.volume
        for Q in range(self.qq):
            for G1,G2 in product(range(self.GG),repeat=2):
                #Find indices of g1=g_Q+G1 and g2=g_Q+G2
                gtemp1=self.gvectors[G1]+self.sgvectors[self.Qqg[Q,2]]
                gtemp2=self.gvectors[G2]+self.sgvectors[self.Qqg[Q,2]]
                #Check that g1 and g2 are not outside the cutoff
                if self.Gcond(gtemp1)==True and self.Gcond(gtemp2)==True:
                    #I'm not proud of this. point_matching does not work
                    #with one-element lists to compare...
                    gtemp1 = np.array([gtemp1,gtemp1])
                    gtemp2 = np.array([gtemp2,gtemp2])
                    _g1 =point_matching(self.sgvectors,gtemp1,double_check=False)
                    g1=_g1[0]
                    _g2 =point_matching(self.sgvectors,gtemp2,double_check=False)
                    g2=_g2[0]
                    #Find indices of iq/iQ, the equivalents of q/Q ind the IBZ
                    iq,_ = self.make_irr(self.isqpts_ind,self.Qqg[Q,1],self.sqpts[self.Qqg[Q,1]],self.isqpts)
                    iQ,_ = self.make_irr(self.iqpts_ind,self.Qqg[Q,0],self.qpts[self.Qqg[Q,0]],self.iqpts)
                    #
                    new_x[iq,g1,g2]=self.ychi.X[iQ,G1,G2]
        #new_x = v_scale*new_x
        return new_x
    def Gcond(self,g_to_check):
        Gmax_values = []
        for i in range(3):
            M = abs(np.max(self.gvectors[:,i]))
            m = abs(np.min(self.gvectors[:,i]))
            if M>=m: Gmax_values.append(M)
            else:    Gmax_values.append(m)
        Gmax_values = np.array(Gmax_values)
        g_tc=abs(g_to_check)
        eps=1e-5
        condition = ( g_tc[0]<=Gmax_values[0]+eps and g_tc[1]<=Gmax_values[1]+eps and g_tc[2]<=Gmax_values[2]+eps )
        return condition
