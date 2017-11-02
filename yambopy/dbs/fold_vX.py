# 1st version by Fulvio Paleari
# This file is part of yambopy
#
# This is the plan:
# (1) 
#           Each Q in uc is connected to a q in sc by a sc lattice vector g_Q.
#           The set of g_Qs allows us to pass from {QG} to {qg} and then from
#           {Q+G1,Q+G2} to {q+g1,q+g2}: we have Q=q+g_Q, g1=g_Q+G1, g2=g_Q+G2.
# (2) 
#           This mapping works in the expanded BZ. This means we have to expand
#           X_uc as well. For now we have to run the uc calculation in the full 
#           cell (noinv, nosym). I have to implement the transformation with the
#           inverse transpose of the symmetry operators to make it work in the IBZ.
# (3)
#           The checks of matching between the (Q+G)- and (q+g)-grids are awful:
#           inefficient and can totally fail. I need to find a smarter and more
#           robust way to do that.
#
from yambopy import *
from itertools import product
import math
import bisect
#
class fold_vX():
    """Starting points: computed vX in full unit cell, trash run of vX in supercell
       with consistent {qG} structure ([1] k-point mesh, [2] blocks cutoff).
    """
    def __init__(self,uc_lattice_path,uc_screening_path,sc_lattice_path,sc_screening_path,out_path='./em1s_folded',check_grids=True):
        # (0) read dbs
        ylat  = YamboLatticeDB(save=uc_lattice_path,expand=False)
        ychi  = YamboStaticScreeningDB(save=uc_screening_path)
        yslat = YamboLatticeDB(save=sc_lattice_path)
        yschi = YamboStaticScreeningDB(save=sc_screening_path)
        self.ychi  = ychi
        self.yschi = yschi
        # now I have Gvectors, sgvectors, iQpts, isqpts in cartesian coordinates
        self.gvectors, self.sgvectors  = ychi.gvectors, yschi.gvectors
        self.iqpts,    self.isqpts     = ychi.qpoints,  yschi.qpoints
        # (1) expand uc and sc qpoints in full BZ
        #uc already expanded
        self.qpts = self.iqpts
        self.sqpts, self.isqpts_ind, self.in_sIBZ = self.expand_qpts(yslat,self.isqpts)
        #
        self.GG, self.sGG   = len(self.gvectors), len(self.sgvectors)
        self.qq, self.iqq   = len(self.qpts),     len(self.iqpts)
        self.sqq,self.isqq  = len(self.sqpts),    len(self.isqpts)
        if check_grids:
            print('Checking coverage...')
            self.check_coverage2()
        # (2) Qqg contains the following indices: (Q,q,g_Q) in {qpts,sqpts,sgvectors}
        print('Creating Qqg list...')
        self.Qqg  = self.get_Qqg()
        print('Qqg list created. Creating Ggg list...')
        self.Ggg  = self.get_Ggg()
        print('Ggg list created. Starting X loop.')
        # (3) Now match the values from X(iQ+G1,iQ+G2) to X(q+g1,q+g2)
        self.X  = self.match_X()
        # (4) Select only X(iq+g1,iq+g2)
        self.new_x = np.array([ self.X[q] for q in range(self.sqq) if self.in_sIBZ[q]==1 ])
        # (5) Save new em1s database. It has the same serial number as the sc database.
        self.yschi.saveDBS(out_path,new_X=self.new_x)
        print('folded DBS saved')
        ###
        ###
    def expand_qpts(self,lat,irr_qpts):
        #[i] Expand qpoints and keep track of the corresponding ones in the IBZ
        lat.expandKpoints(qpoints=irr_qpts)
        qpts = lat.car_kpoints
        iqpts_ind = lat.kpoints_indexes
        map_irr_to_full = point_matching(qpts,irr_qpts,double_check=False)
        in_IBZ = np.zeros([len(qpts)],dtype=np.int)
        in_IBZ[map_irr_to_full]=1
        return qpts, iqpts_ind, in_IBZ
        #
    def expand_QG(self,q_points,g_vectors):
        #[ii] Create expanded list of q+g vectors with index mapping
        QQ, GG = len(q_points), len(g_vectors)
        exp_pts   = np.zeros([QQ*GG,3])
        exp_to_uc = np.zeros([QQ*GG,2],dtype=np.int64)
        uc_to_exp = np.zeros([QQ,GG],dtype=np.int64)
        for q,g in product(range(QQ),range(GG)):
            it = q*GG+g
            exp_pts[it]   = q_points[q]+g_vectors[g]
            exp_to_uc[it] = [q,g]
            uc_to_exp[q,g]= it 
        return exp_pts,exp_to_uc,uc_to_exp
        #
    def check_coverage(self):
        exp_uc,_,_ = self.expand_QG(self.qpts,self.gvectors)
        exp_sc,exp_to_sc,_ = self.expand_QG(self.sqpts,self.sgvectors)
        count = 0
        list_of_points_not_covered = []
        for i,qp in enumerate(exp_sc):
            for Qp in exp_uc:
                if np.isclose(qp,Qp,rtol=1e-05,atol=1e-05).all(): 
                    count+=1
                    list_of_points_not_covered.append(exp_to_sc[i])
        if count != 0:
            list_of_points_not_covered = np.array(list_of_points_not_covered)
            np.savetxt('points_not_covered.dat',list_of_points_not_covered) 
            raise IOError("[ERROR] %d qpoints not covered. Try to increase NGsBlkX in the unit cell calculation."%count)
        else: print('Ok')
    def check_coverage2(self,n=5):
        #[iii] Check that the number of blocks in the uc covers all the sc points
        exp_uc,_,_ = self.expand_QG(self.qpts,self.gvectors)
        exp_sc,exp_to_sc,_ = self.expand_QG(self.sqpts,self.sgvectors)
        ro_uc, ro_sc = np.round(exp_uc,n), np.round(exp_sc,n)
        ro_uc, ro_sc = ro_uc.tolist(), ro_sc.tolist()
        count = 0
        list_of_points_not_covered = []
        for i,g in enumerate(ro_sc):
            if g not in ro_uc: 
                count+=1
                list_of_points_not_covered.append(exp_to_sc[i])
        if count!=0: 
            list_of_points_not_covered = np.array(list_of_points_not_covered)
            np.savetxt('points_not_covered.dat',list_of_points_not_covered)            
            raise IOError("[ERROR] %d qpoints not covered. Try to increase NGsBlkX in the unit cell calculation."%count)
        else: print('Ok')   
        #      
    def get_Qqg(self,thr=1e-6):
        #[iv] Get the {Q,q,g_Q} index list [N^3 loop. I guess this could be improved?]
        qtemp = np.array( [[Q-q for q in self.sqpts] for Q in self.qpts] )
        Qqg_list = []
        for i,Q in enumerate(qtemp):
            for j,q in enumerate(Q):
                for k,g in enumerate(self.sgvectors):
                    if (abs(q-g)<thr).all(): 
                        Qqg_list.append([i,j,k])
        Qqg_list = np.array(Qqg_list)
        return Qqg_list
        #
    def get_Ggg(self):
        #[v] Get the {G,g_Q,g} index list
        g_Q_ind = np.unique(self.Qqg[:,2])
        g_Qs    = np.array( [ self.sgvectors[i] for i in g_Q_ind ])
        g_list = np.zeros([self.GG,len(g_Q_ind),3])
        g_indices = np.zeros([self.GG,len(g_Q_ind)],dtype=np.int64)
        for G in range(self.GG): 
            g_list[G]=self.gvectors[G]+g_Qs
            g_indices[G] = point_matching(self.sgvectors,g_list[G],double_check=False)
        Ggg = []
        for G,gq in product(range(self.GG),range(len(g_Q_ind))):
            #This call is the heaviest part of the script: it should be made faster somehow 
            if self.vecs_find(g_list[G,gq])==True: Ggg.append([G,g_Q_ind[gq],g_indices[G,gq]])
        Ggg = np.array(Ggg)
        return Ggg
        #
    def vecs_find(self,vec1):
        check = False
        for g in self.sgvectors:
            if np.isclose(g,vec1,rtol=1e-05,atol=1e-05).all(): check = True 
        return check
    def vecs_find2(self,vec1,n=6):
        #[vi] Check if g-vectors in Gg[g] list are inside the boundaries
        ro_sc, ro_1  = np.round(self.sgvectors,n), np.round(vec1,n)
        ro_sc, ro_1  = ro_sc.tolist(), ro_1.tolist()
        return ro_1 in ro_sc
    def match_X(self):
        #[vii] Loop on Q,G1,G2 (~N^3) to obtain X(q)_g1g2
        gvectors  = self.gvectors
        sgvectors = self.sgvectors
        full_x = np.zeros([self.sqq,self.sGG,self.sGG],dtype=np.complex64)
        for Q in range(self.qq):
            g_Q = self.Qqg[Q,2]
            q   = self.Qqg[Q,1]
            G_and_g = self.Ggg[self.Ggg[:,1]==g_Q]
            for j1,j2 in product(range(len(G_and_g)),repeat=2):
            #for j1 in range(len(G_and_g)):
                #j2=j1
                G1,G2 = G_and_g[j1,0], G_and_g[j2,0]
                g1,g2 = G_and_g[j1,2], G_and_g[j2,2]
                full_x[q,g1,g2]=self.ychi.X[Q,G1,G2] 
        return full_x
