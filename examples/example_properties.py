"""
Calculate Excitation Energies and Couplings Via TDDFT
From: "Steering charge-transfer state formation in strongly-coupled dimers using DNA scaffolds"
       (in preparation)

Author: Maria Castellanos
Willard Group, MIT

"""
import sys
import time
start_time = time.time()

import numpy as np
import h5py
import scipy.linalg
import math
from pyscf import lib

#MD analysis tools
import MDAnalysis
from csv import reader

#All functions to calculate coupling
import couplingutils as cp

#Defining relevant paramters
H_to_eV = 27.211 #All couplings are in a.u.
H_to_cm = 219474.6  #All couplings are in a.u.
basis = "6-31g"
xc = "b3lyp"
ch = 0
scf_cycles = 300

##Importing MD trajectories

param_file =  "file.prmtop" # Change to file name 
traj_file = "file.nc" # Change to trajectory file (or .pdb file) 
print("Traj file is ",traj_file)
f_format = "TRJ" # or "PDB", MDanalysis can also recongnize format automatically.
if f_format == "PDB":
    u = MDAnalysis.Universe(traj_file, format=f_format)
else:
    u = MDAnalysis.Universe(param_file, traj_file)
tin = 0 # starting frame
dt = 2 # time step
md_dt = 10 #time step of imported MD trajectory (for printing only)

resnum1 = [185,186,187] # List of residue ids on molecule to isolate
resnum2 = [230,231,232]

# A list of atoms to delete from residues above (per residue in resnum). Then, a list of the atoms to be capped with H
del_list = [[0,['C5','C6','O3','H9','H10','H11','H12','P','O4','O5'],['O2']],
            [1,[''],['']], 
            [2,['C5','C6','O3','H9','H10','H11','H12'],['O2']]] 
# Optional: Coordinates of atom where we want H to be placed. 
cap_pos1 = [[0,[(186,'P')]], [], [2,[(189,'P')]]] #The H will be placed in the position of P of res 185 & of P of res 189 
cap_pos2 = [[0,[(230,'P')]], [], [2,[(233,'P')]]] 


a_save = []
b_save = []
c_save = []
d_save = []

istep = int(sys.argv[1]) # optional python input useful for splitting long trajectories

num_sol = (len(u.trajectory)
save_path = "/path/to/save/data/"
print("Step #" + str(istep), "total: ", num_sol)


for ts in u.trajectory[num_sol*(istep-1):num_sol*istep:dt]:

    print("BEFORE: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #DFT
    xyzA,xyzB,RAB = cp.Process_MD(u,resnum1,resnum2,del_list=del_list,cap_list=[cap_pos1,cap_pos2])
    # generates a pdb to ckeck isolated molecule
    test = cp.pdb_cap(u,resnum1,resnum2,del_list=del_list,path_save=save_path+'pdb_molecule.pdb',MDA_selection='all',cap_list=[cap_pos1,cap_pos2])
    molA,mfA,o_A,v_A = cp.do_dft(xyzA, basis=basis, xc_f=xc, mol_ch=ch, spin=1,
                                 verb=3,scf_cycles=scf_cycles,opt_cap=None)
    molB,mfB,o_B,v_B = cp.do_dft(xyzB, basis=basis, xc_f=xc, mol_ch=ch, spin=1,
                                 verb=3,scf_cycles=scf_cycles,opt_cap=None)

    print("AFTER dft: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))
    #TDDFT
    TenA, TdipA, tdmA = cp.do_tddft(mfA,o_A,v_A,0)
    TenB, TdipB, tdmB = cp.do_tddft(mfB,o_B,v_B,0)   

    ## Oscillator strength
    O_stA = TenA * np.linalg.norm(TdipA)**2
    O_stB = TenB * np.linalg.norm(TdipB)**2

    print("AFTER TDDFT: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))
    #Calculating V Coulombic
    __, chgA = cp.td_chrg_lowdin(molA, tdmA)
    __, chgB = cp.td_chrg_lowdin(molB, tdmB)

    Vtdipole = cp.V_multipole(molA,molB,chgA,chgB) #using monopole approx

    V_Cfull = cp.V_Coulomb(molA, molB, tdmA, tdmB, calcK=False)
        
    print("AFTER VCoulomb: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))

    #Calculating V CT
    te,th = cp.transfer_CT(molA,molB,o_A,o_B,v_A,v_B) #transfer integrals
    mfAB = TenAB = 0
    VCT,dw,Rab = cp.V_CT(te,th,mfAB,TenAB,RAB)
    print("AFTER VCT: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))
    
    t_i = round((ts.frame*md_dt + tin),2)

    save_V = [t_i, V_Cfull*H_to_cm, VCT*H_to_cm, Rab, Vtdipole]
    save_t = [t_i, te*H_to_cm, th*H_to_cm, dw*H_to_cm]
    save_abs = [TenA*H_to_cm, TenB*H_to_cm, O_stA, O_stB]
    save_tdm = list(np.append(TdipA,TdipB))

    a_save.append(save_V)
    b_save.append(save_t)
    c_save.append(save_abs)
    d_save.append(save_tdm)

    # Create target Directory if it doesn't exist
    label = traj_file.replace('.nc', '')
    if not os.path.exists(save_path+label):
        os.mkdir(save_path+label)
        print("Directory ", label,  " Created ")
    else:    
        print("Directory ", label,  " already exists")

    i_step = str(istep)
    np.savetxt(save_path+label+"/couplings_"+i_step+".txt", np.array(a_save), fmt=['%.1f','%1.4e','%1.4e','%.4f','%1.4e'])
    np.savetxt(save_path+label+"/tintegrals_"+i_step+".txt", np.array(b_save), fmt=['%.1f','%1.4e','%1.4e','%1.4e'])
    np.savetxt(save_path+label+"/absspec_"+i_step+".txt", np.array(c_save), fmt='%1.4e')
    np.savetxt(save_path+label+"/tdm_"+i_step+".txt", np.array(d_save), fmt='%1.4e') 

