"""
Calculate Excitation Energies and Couplings Via TDDFT
From: "Activating charge-transfer state formation in strongly-coupled dimers using DNA scaffolds" 

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
import dyeScreen.commons.couplingutils as cp

#Defining relevant paramters
H_to_eV = 27.211 #All couplings are in a.u.
H_to_cm = 219474.6  #All couplings are in a.u.
basis = "6-31g"
xc = "b3lyp"
ch = 0 #total charge of the molecule
scf_cycles = 300 #increase if SCF doesn't converge

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

resnum1 = [21,22] # (example) List of residue ids on molecule to isolate
resnum2 = [31,32] # Resnum1 and resnum2 are identical  molecules

# A list of atoms to delete from residues above (per residue in resnum). Then, a list of the atoms to be capped with H
del_list = [[0,['C1','O2','H5','P'],['O1']], # C1, O2, H5 and P will be deleted from res 21 (and 31). O1 will be capped with H 
            [1,['C2','O3','H4'],['O2']]]  

V_save = []
a_save = []

istep = int(sys.argv[1]) # optional python input useful for splitting long trajectories

num_sol = (len(u.trajectory)
save_path = "/path/to/save/data/"

for ts in u.trajectory[num_sol*(istep-1):num_sol*istep:dt]:

    print("BEFORE: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #DFT
    xyzA,xyzB,RAB = cp.Process_MD(u,resnum1,resnum2,del_list=del_list)
    # generates a pdb to ckeck isolated molecule
    test = cp.pdb_cap(u,resnum1,resnum2,del_list=del_list,path_save=save_path+'pdb_molecule.pdb',MDA_selection='all')

    molA,mfA,o_A,v_A = cp.do_dft(xyzA, basis=basis, xc_f=xc, mol_ch=ch, spin=1,
                                 verb=3,scf_cycles=scf_cycles)
    molB,mfB,o_B,v_B = cp.do_dft(xyzB, basis=basis, xc_f=xc, mol_ch=ch, spin=1,
                                 verb=3,scf_cycles=scf_cycles)

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

    Vmpole = cp.V_monopole(molA,molB,chgA,chgB) #using monopole approx

    V_Cfull = cp.V_Coulomb(molA, molB, tdmA, tdmB, calcK=False)
        
    print("AFTER VCoulomb: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))

    #Calculating V CT
    te,th = cp.transfer_CT(molA,molB,o_A,o_B,v_A,v_B) #transfer integrals
    mfAB = TenAB = 0 # ommiting dimer DFT
    VCT,dw,Rab = cp.V_CT(te,th,mfAB,TenAB,RAB)
    print("AFTER VCT: ",lib.num_threads())
    print("--- %s seconds ---" % (time.time() - start_time))
    
    t_i = round((ts.frame*md_dt + tin),2)

    save_V = [t_i, V_Cfull*H_to_cm, VCT*H_to_cm, Vmpole*H_to_cm]
    save_abs = [TenA*H_to_cm, TenB*H_to_cm, O_stA, O_stB] # Useful for absorption spectra prediction

    V_save.append(save_V)
    a_save.append(save_abs)

    # Create target Directory if it doesn't exist
    label = traj_file.replace('.nc', '')
    if not os.path.exists(save_path+label):
        os.mkdir(save_path+label)
        print("Directory ", label,  " Created ")
    else:    
        print("Directory ", label,  " already exists")

    i_step = str(istep)
    np.savetxt(save_path+label+"/couplings_"+i_step+".txt", np.array(V_save), fmt=['%.1f','%1.4e','%1.4e','%.4f'])
    np.savetxt(save_path+label+"/absspec_"+i_step+".txt", np.array(a_save), fmt='%1.4e')

