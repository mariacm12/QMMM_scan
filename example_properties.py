"""
QMM_traj.py

Author: Maria Castellanos
Willard Group, MIT

Calculate Excitation Energies and Couplings Via TDDFT

"""
import sys
import time
start_time = time.time()

import numpy as np
from pyscf import lib

#MD analysis tools
import MDAnalysis
#All functions to calculate coupling
import couplingutils as cp

#Defining relevant paramters
H_to_eV = 27.2114 #All couplings are in a.u.
H_to_cm = 219474.6
basis = "6-31g" # basis set
xc = "b3lyp" # exchange func
ch = 0 # Molecule's total charge 

##Importing MD trajectories

# Repetition index and trajectory file taken as input
rep = ["A","B","D","E"][int(sys.argv[2])]
traj = int(sys.argv[1]) # File index (A single traj is divided in multiple files).


param = "MD/SqOpp_dimer.prmtop"
traj_prefix =  "MD/prod/SqOppDim_prod"
u = MDAnalysis.Universe(param, traj_prefix+rep+str(traj)+".nc", format="TRJ")

dt = 2 # Do QM calculates every dt MD frames
MD_dt = 10 # Actual time step in ps (from MD trajectories)

# res ID of the dimer molecules
sel1 = '11'
sel2 = '32'

a_save = []
b_save = []
c_save = []
d_save = []
e_save = []


num_sol = len(u.trajectory)
extra_t = 0
save_path = "QM_results/"
for ts in u.trajectory[0:num_sol:dt]:

    print("BEFORE: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))
    #DFT
    xyzA,xyzB,RAB = cp.Process_MD(u,sel1,sel2,coord_path='coord_files/MD_atoms')
    molA,mfA,o_A,v_A = cp.do_dft(xyzA,basis=basis,xc_f=xc,mol_ch=ch,spin=1,verb=3)
    molB,mfB,o_B,v_B = cp.do_dft(xyzB,basis=basis,xc_f=xc,mol_ch=ch,spin=1,verb=3)

    print("AFTER dft: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #TDDFT
    TenA, TdipA, tdmA = cp.do_tddft(mfA,o_A,v_A,0)
    TenB, TdipB, tdmB = cp.do_tddft(mfB,o_B,v_B,0)   

    # Oscillator strength
    O_stA = TenA * np.linalg.norm(TdipA)**2
    O_stB = TenB * np.linalg.norm(TdipB)**2

    print("AFTER TDDFT: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))
    #Calculating V Coulombic
    __, chgA = cp.td_chrg_lowdin(molA, tdmA)
    __, chgB = cp.td_chrg_lowdin(molB, tdmB)
    Vtdipole = cp.V_multipole(molA,molB,chgA,chgB) #using t monopole approx

    Vpdipole = cp.V_pdipole(TdipA,TdipB,RAB)*H_to_eV #using the point-dipole approx

    V_Cfull = cp.V_Coulomb(molA, molB, tdmA, tdmB, calcK=False)
    print("AFTER VCoulomb: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))

    #Calculating V CT
    te,th = cp.transfer_CT(molA,molB,o_A,o_B,v_A,v_B)

    # dimer DFT (***commented because it takes too long with this size of molecule)
    #molAB,mfAB,o_AB,v_AB = cp.dimer_dft(molA,molB,xc_f=xc,verb=3)
    #TenAB, TdipAB, __ = cp.do_tddft(mfAB, o_AB, v_AB, state_id=list(range(7)))
    # If we were to calculate te/th for a symmetric mol
    #tes,ths = cp.transfer_sym(mfAB) 
    #VCT_sym,__ = cp.V_CT(tes,ths,mfAB,TenAB,RAB)

    VCT,dw,Rab = cp.V_CT(te,th, RAB)

    print("AFTER VCT: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))


    t_i = round((ts.frame*MD_dt),2)

    save_V = [t_i, V_Cfull*H_to_cm, VCT*H_to_cm, Rab, Vtdipole*H_to_cm]
    save_t = [t_i, te*H_to_cm, th*H_to_cm, dw*H_to_cm]
    save_abs = [TenA*H_to_cm, TenB*H_to_cm, O_stA, O_stB]
    save_tdm = list(np.append(TdipA,TdipB))

    a_save.append(save_V)
    b_save.append(save_t)
    c_save.append(save_abs)
    d_save.append(save_tdm)

    np.savetxt(save_path+"t_"+rep+str(traj)+"/couplings.txt", np.array(a_save), fmt=['%.1f','%1.4e','%1.4e','%.4f','%1.4e'])
    np.savetxt(save_path+"t_"+rep+str(traj)+"/tintegrals.txt", np.array(b_save), fmt=['%.1f','%1.4e','%1.4e','%1.4e'])
    np.savetxt(save_path+"t_"+rep+str(traj)+"/absspec.txt", np.array(c_save), fmt='%1.4e')
    np.savetxt(save_path+"t_"+rep+str(traj)+"/tdm.txt", np.array(d_save), fmt='%1.4e')
