#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupling functions for DFT calculation

Created on Mon Apr 12 15:55:49 2021

@author: mariacm
"""

import numpy as np
import h5py
import scipy.linalg
from pyscf import gto, scf, tdscf, lib, dft, lo, solvent
from functools import reduce

#MD analysis tools
import MDAnalysis
from csv import reader

# =============================================================================
# Coulombic coupling (based on code by Ardavan Farahvash and Qiming Sun)
# =============================================================================

def td_chrg_lowdin(mol, dm):
    """
    Calculates Lowdin Transition Partial Charges
    
    Parameters
    ----------
    mol. PySCF Molecule Object
    dm. Numpy Array. Transition Density Matrix in Atomic Orbital Basis
    
    Returns
    -------
    pop. Numpy Array. Population in each orbital.
    chg. Numpy Array. Charge on each atom.
    """
    #Atomic Orbital Overlap basis
    s = scf.hf.get_ovlp(mol)
    
    U,s_diag,_ = np.linalg.svd(s,hermitian=True)
    S_half = U.dot(np.diag(s_diag**(0.5))).dot(U.T)
    
    pop = np.einsum('ij,jk,ki->i',S_half, dm, S_half)

    print(' ** Lowdin atomic charges  **')
    chg = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        chg[s[0]] += pop[i]
        
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        print('charge of  %d%s =   %10.5f'%(ia, symb, chg[ia]))
    
    return pop, chg

def jk_ints_eff(molA, molB, tdmA, tdmB, calcK=False):
    """
    A more-efficient version of two-molecule JK integrals.
    This implementation is a bit blackbox and relies and calculation the HF
    potential before trying to calculate couplings. 

    Parameters
    ----------
    molA/molB : PySCF Mol Obj. Molecule A and Molecule B.
    tdmA/tdmB : Numpy Array. Transiiton density Matrix

    Returns
    -------
    cJ ~ Coulomb Coupling
    cK ~ Exchange Coupling
    
    V_{ab} = 2J - K
    """
    
    from pyscf.scf import jk, _vhf
    naoA = molA.nao
    naoB = molB.nao
    assert(tdmA.shape == (naoA, naoA))
    assert(tdmB.shape == (naoB, naoB))

    molAB = molA + molB
    
    #vhf = Hartree Fock Potential
    vhfopt = _vhf.VHFOpt(molAB, 'int2e', 'CVHFnrs8_prescreen',
                         'CVHFsetnr_direct_scf',
                         'CVHFsetnr_direct_scf_dm')
    dmAB = scipy.linalg.block_diag(tdmA, tdmB)
    #### Initialization for AO-direct JK builder
    # The prescreen function CVHFnrs8_prescreen indexes q_cond and dm_cond
    # over the entire basis.  "set_dm" in function jk.get_jk/direct_bindm only
    # creates a subblock of dm_cond which is not compatible with
    # CVHFnrs8_prescreen.
    vhfopt.set_dm(dmAB, molAB._atm, molAB._bas, molAB._env)
    # Then skip the "set_dm" initialization in function jk.get_jk/direct_bindm.
    vhfopt._dmcondname = None
    ####

    # Coulomb integrals
    with lib.temporary_env(vhfopt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        shls_slice = (0        , molA.nbas , 0        , molA.nbas,
                      molA.nbas, molAB.nbas, molA.nbas, molAB.nbas)  # AABB
        vJ = jk.get_jk(molAB, tdmB, 'ijkl,lk->s2ij', shls_slice=shls_slice,
                       vhfopt=vhfopt, aosym='s4', hermi=1)
        cJ = np.einsum('ia,ia->', vJ, tdmA)
        
    if calcK==True:
        # Exchange integrals
        with lib.temporary_env(vhfopt._this.contents,
                               fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
            shls_slice = (0        , molA.nbas , molA.nbas, molAB.nbas,
                          molA.nbas, molAB.nbas, 0        , molA.nbas)  # ABBA
            vK = jk.get_jk(molAB, tdmB, 'ijkl,jk->il', shls_slice=shls_slice,
                           vhfopt=vhfopt, aosym='s1', hermi=0)
            cK = np.einsum('ia,ia->', vK, tdmA)
            
        return cJ, cK
    
    else: 
        return cJ, 0

# =============================================================================
# Functions to calculate QM properties
# =============================================================================
def V_Coulomb(molA, molB, tdmA, tdmB, calcK=False):
    '''
    Full coupling (slower, obviously)
    Parameters
    ----------
    molA/molB : PySCF Mol Obj. Molecule A and Molecule B.
    tdmA/tdmB : Numpy Array. Transiiton density Matrix
    calcK : Boolean, optional
       Whether to calculate exchange integral. The default is False.

    Returns
    -------
    Coulombic coupling, Vij.

    '''
    cJ,cK = jk_ints_eff(molA, molB, tdmA, tdmB, calcK=False)
    return 2*cJ - cK

def V_multipole(molA,molB,chrgA,chrgB):
    """
    Coupling according to the multiple approximation
    
    Parameters
    ----------
    molA : pyscf mol
        molecule A.
    molB : pyscf mol
        molecule B.
    chrgA : ndarray
    chrgB : ndarray

    Returns
    -------
    Vij : float
        The Coulombic coupling in the monpole approx

    """
    
    from scipy.spatial.distance import cdist,pdist
    
    mol_dist = cdist(molA.atom_coords(),molB.atom_coords()) 
    Vij = np.sum( np.outer(chrgA,chrgB)/mol_dist ) #SUM_{f,g}[ (qf qg)/|rf-rg| ]

    return Vij

def V_pdipole(td1,td2,rAB):
    
    const = 1 #a.u.
    rAB *= 1.8897259886
    miuAnorm = abs(np.linalg.norm(td1))
    miuBnorm = abs(np.linalg.norm(td2))
    RABnorm = np.linalg.norm(rAB)
    num = np.dot(td1,td2)-3*np.dot(td1,rAB)*np.dot(td2,rAB)

    return (miuAnorm*miuBnorm/const)*num/RABnorm**3

def transfer_CT(molA,molB,o_A,o_B,v_A,v_B):
    '''
    Calculating the electron/hole transfer integrals from 1e- overlap matrix elements

    Parameters
    ----------
    molA, molB : Pyscf HF molecules
    o_A, o_B  : ndarray
        Occupied orbitals.
    v_A,v_B : ndarray
        Virtual orbitals.

    Returns
    -------
    te : float
        Electron transfer integral.
    th : float
        Hole transfer integral.

    '''
    from pyscf import ao2mo

    naoA = molA.nao #of atomic orbitals in molecule A
    
    #1 electron integrals between molA and molB, AO basis
    eri_AB = gto.intor_cross('int1e_ovlp',molA,molB) 
    # Transform integral to from AO to MO basis
    eri_ab = lib.einsum('pq,pa,qb->ab', eri_AB, v_A, v_B) #virtual
    eri_ij = lib.einsum('pq,pi,qj->ij', eri_AB, o_A, o_B) #occupied

    te = eri_ab[0][0]
    th = -eri_ij[-1][-1]

    print("**transfer integrals=",te,th)
    print(eri_ab[0][0],eri_ab[-1][-1])
    print(eri_ij[0][0],eri_ij[-1][-1])
    return te,th

def V_CT(te,th,mf,Egap,rab):
    """
    CT coupling

    Parameters
    ----------
    te, th : float
        electron/hole transfer int.
    mfA, mfB : Mean-field PySCF objects
    Egap : float
        Transition energy from TDDFT
    rab : ndarray
        center of mass distance vector between chromophores.

    Returns
    -------
    float
        CT coupling.

    """
    RAB = np.linalg.norm(rab)*1.8897259886 #Ang to a.u.
   
    #Energy of frontier orbitals
    #EL = mf.mo_energy[mf.mo_occ==0][0]
    #EH = mf.mo_energy[mf.mo_occ!=0][-1]

    #Fundamental gap
    #Eg = EL - EH
    #optical gap
    #Eopt = Egap
    #Local Binding energy
    #U = Eg - Eopt 
    U = 0.7*0.0367493
    #Coulomb Binding energy
    perm = 1 #4*pi*e_0 in a.u.
    er = 77.16600 #water at 301.65K and 1 bar
    
    elect = 1 #charge of e-
    V = elect**2/(perm*er*RAB)
    domega = U-V

    return -2*te*th/(domega), domega, np.linalg.norm(rab)



def transfer_sym(mf):
    '''
    e- transfer integrals assuming dimer is symmetric

    Parameters
    ----------
    molA,molB : Pyscf molecule

    Returns
    -------
    te : float
        e- transfer integral
    th : float
        h+ transfer integral

    '''
    
    #MOs for the dimer

    mo_en = mf.mo_energy
    E_v = mo_en[mf.mo_occ==0]
    E_o = mo_en[mf.mo_occ!=0]
    #Frontier Energies
    EH,EHm1 = E_o[-1],E_o[-2]
    EL,ELp1 = E_v[0],E_v[1]
    
    #transfer integrals
    th = (EH-EHm1)/2
    te = (EL-ELp1)/2
    
    return te,th

def dimer_dft(molA,molB,xc_f='b3lyp',verb=4):
    mol = molA+molB
    mol.verbose = verb
    mf = scf.RKS(mol)
    mf.xc= xc_f
    #Run with COSMO implicit solvent model
    mf = solvent.ddCOSMO(mf).run()#mf.run()
    
    mo = mf.mo_coeff #MO Coefficients
    occ = mo[:,mf.mo_occ!=0] #occupied orbitals
    virt = mo[:,mf.mo_occ==0] #virtual orbitals   

    return mol,mf,occ,virt

def do_dft(coord,basis='6-31g',xc_f='b3lyp',mol_ch=0,spin=0,verb=4):
        
    #Make Molecule Object

    #Make SCF Object, Diagonalize Fock Matrix
    mol = gto.M(atom=coord[1:-1],basis=basis,charge=mol_ch,spin=0)
    mol.verbose = verb

    mf = scf.RKS(mol)
    mf.xc= xc_f
    #Run with COSMO implicit solvent model
    mf = solvent.ddCOSMO(mf).run()#mf.run()
    
    mo = mf.mo_coeff #MO Coefficients
    occ = mo[:,mf.mo_occ!=0] #occupied orbitals
    virt = mo[:,mf.mo_occ==0] #virtual orbitals   

    return mol,mf,occ,virt

def do_tddft(mf,o_A,v_A,state_id=0):
    """  

    Parameters
    ----------
    mf : pyscf scf object
        result from DFT
    o_A : ndarray
        Occupied orbitals
    v_A : ndarray
        Virtual orbitals
    state_ids : list 
        Wanted excitated states 
        (e.g., [0,1] returns fro 1st and 2nd exc states)

    Returns
    -------
    None.

    """
    nstates = 1 if isinstance(state_id,int) else len(state_id)    
    td = mf.TDA().run(nstates=nstates) #Do TDDFT-TDA

    if isinstance(state_id,list):
        Tenergy = [td.e[i] for i in state_id]
        Tdipole = [td.transition_dipole()[i] for i in state_id]
    else:
        Tdipole = td.transition_dipole()[state_id]
        Tenergy = td.e[state_id]

    # The CIS coeffcients, shape [nocc,nvirt]
    # Index 0 ~ X matrix/CIS coefficients, Index Y ~ Deexcitation Coefficients
    cis_A = td.xy[0][0] #[state_id][0]
 
    #Calculate Ground to Excited State (Transition) Density Matrix
    tdm = np.sqrt(2) * o_A.dot(cis_A).dot(v_A.T)
    
    return Tenergy, Tdipole, tdm


def Process_MD(u,sel_1,sel_2,coord_path='/coord_files/MD_atoms'):
    """
    

    Parameters
    ----------
    u   : MDAnalysis universe
        Object containing MD trajectory
    sel_1 : str
        String with the residue name of Molecule A.
    sel_2 : str
        String with the residue name of Molecule B.

    Returns
    -------
    None.

    """
    frameN = u.trajectory.frame
    agA = u.select_atoms("resid "+str(sel_1))
    agB = u.select_atoms("resid "+str(sel_2))
    
    resA = agA.resids[0]
    xyzA = agA.positions
    namesA = agA.atoms.names
    CofMA = agA.center_of_mass()
    
    resB = agB.resids[0]
    xyzB = agB.positions
    namesB= agB.atoms.names
    CofMB = agB.center_of_mass()

    #Adding Hs
    def add_H(xyz,names,sel):
        #atoms to delete
        op1 = np.nonzero(names=='OP1')[0][0]
        op2 = np.nonzero(names=='OP2')[0][0]
        p = np.nonzero(names=='P')[0][0]
        xyz_new = np.delete(xyz,[op1,op2,p],0)
        names_new = np.delete(names,[op1,op2,p],0)
        
        #capping with H
        O3 = u.select_atoms("resid "+str(sel)+" and name O3*")
        O5 = u.select_atoms("resid "+str(sel)+" and name O5*")
        O3_coord = O3.positions + 0.6
        O5_coord = O5.positions + 0.6
        xyz_add = np.append(xyz_new,O3_coord,axis=0)
        xyz_add = np.append(xyz_add,O5_coord,axis=0)
        names_add = np.append(names_new,['H']*2,axis=0)
        return names_add,xyz_add
    
    
    #Storing coordinates in csv file
    def coord_save(resN,xyz,names):
        text = np.array([str(names[i] + ' ' +
                             np.array2string(xyz[i],precision=6, separator="i").replace('[','').replace(']','')).replace('O5\'','O5*').replace('O3\'','O3*').replace('OP1','O^1').replace('OP2','O^2')
                         for i in range(len(names))])
        np.savetxt(coord_path+str(int(resN))+str(int(frameN))+'.csv', [text], delimiter=';', fmt='%s')
    
        with open(coord_path+str(int(resN))+str(int(frameN))+'.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                 xyz_i = np.array2string(np.array(row),precision=6, separator=",").replace('i',',').replace(']','').replace('[','')

        return xyz_i

    namesA_H, xyzA_H = add_H(xyzA,namesA,sel_1)
    namesB_H, xyzB_H = add_H(xyzB,namesB,sel_2)

    coordA = coord_save(resA,xyzA_H,namesA_H)
    coordB = coord_save(resB,xyzB_H,namesB_H)
    
    Rab = CofMA-CofMB    
    return coordA,coordB,Rab

