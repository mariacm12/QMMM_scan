#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupling functions for DFT calculation
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

#For cambrlyp and above
from pyscf.dft import xcfun
#dft.numint.libxc = xcfun
# =============================================================================
# QM functions from J. Chem. Phys. 153, 074111 (2020)
# =============================================================================

def td_chrg_lowdin(mol, dm):
    """
    Calculates Lowdin Transition Partial Charges
    
    Parameters
    ----------
    mol: PySCF Molecule Object
    dm: Numpy Array. Transition Density Matrix in Atomic Orbital Basis
    
    Returns
    -------
    pop: Numpy Array. Population in each orbital.
    chg: Numpy Array. Charge on each atom.
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
    """
    Point dipole approximation
    """
    
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
    U = 0.7*0.0367493 #preset number (can be changed)
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
    mf : Pyscf mean field object

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
    """
    Performs DFT on a dimer given by PySCF objects molA and molB
    """
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

def do_dft(coord,basis='6-31g',xc_f='b3lyp',mol_ch=0,spin=0,verb=4,scf_cycles=200,opt_cap=None):
    """
    Performs DFT calculation for a coordinate array
    """
        
    #Make SCF Object, Diagonalize Fock Matrix
    mol = gto.M(atom=coord,basis=basis,charge=mol_ch,spin=0)
    mol.verbose = verb

    #optimize cap
    if opt_cap is not None:
        mf = scf.RHF(mol)
        mol = contrained_opt(mf, opt_cap)

    mf = scf.RKS(mol)
    mf.xc= xc_f
    mf.max_cycle = scf_cycles
    mf.conv_tol = 1e-6
        
    #mf._numint.libxc = xcfun
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

def cap_H_general(u,sel,sel_id,res_list,H_loc=[]):
    '''
    Will cap a residue with H atoms, given a list of atoms to delete/replace
    
    Parameters
    ----------
    sel : MDAnalysis atom selection
        current atom selection
    sel_id : list
        The residue ids in current selection
    res_list : list
        Includes the list of atoms to delete and those to delete that must be 
        replaced with H, per residue.
        [res_pos ,[atoms_to_del],[atom_to_del_and_cap]]
        - res_pos is with respect to the sel_id list
    H_loc : list, optional
        If provided, determines the the atom to be replaced by H [res_pos,[(resid,atom_name)].
	Must be of same length as res_pos and must give the position for every atom to be capped.
        If not provided, the H position is the same as the capped position, displaced.

    Returns
    -------
    None.

    '''

    all_names = np.empty((0))
    all_types = np.empty((0))
    all_xyz = np.empty((0,3))
    for i in range(len(res_list)):
        r = res_list[i]
        res_i = sel_id[r[0]]
        sel_2 = sel.select_atoms("resid "+str(res_i))
        xyz = sel_2.positions
        names = sel_2.atoms.names
        types = sel_2.atoms.types
 
        #atoms to delete
        try:
            datom_idx = [np.nonzero(names==i)[0][0] for i in r[1]]
            print('***',res_i,datom_idx)
        except:
            datom_idx = []
            print("The atom_name to delete couldn't be found. No atoms will be deleted in res "+str(res_i))
        try:
            cap_idx = [np.nonzero(names==i)[0][0] for i in r[2]]
        except:
            cap_idx = []
            print("The atom_name to cap couldn't be found. No atoms will be replaced with H in res "+str(res_i))
        if len(H_loc)>0 and len(H_loc[i])>0:
            cap_i = H_loc[i][1]
            assert len(cap_i) == len(r[2])
            for ci in cap_i:
                cap_sel = u.select_atoms("resid "+str(ci[0])+" and name "+ci[1])
                cap_pos = cap_sel.positions
        else: 
            cap_pos = xyz[cap_idx] + 0.4
        num_caps = cap_pos.shape[0]
        
        xyz_new = np.delete(xyz,datom_idx,0)
        names_new = np.delete(names,datom_idx,0)
        types_new = np.delete(types,datom_idx,0)

        #capping with H
        xyz_add = np.vstack((xyz_new,cap_pos))
        names_add = np.append(names_new,['H']*num_caps,axis=0)
        types_add = np.append(types_new,['h1']*num_caps,axis=0)

        all_xyz = np.vstack((all_xyz,xyz_add))
        all_names = np.append(all_names,names_add,0)
        all_types = np.append(all_types,types_add,0)
        
    return all_names,all_xyz,all_types

def contrained_opt(mf, constrain_str):
    '''
    Perform constrained optimation given pyscf-formatted constrain string
    '''
    from pyscf.geomopt.geometric_solver import optimize
    #writin the constrains on a file 
    f = open("constraints.txt", "w")
    f.write("$freeze\n")
    f.write("xyz " + constrain_str)
    f.close()

    params = {"constraints": "constraints.txt",}
    mol_eq = optimize(mf, **params)
    mol_eq = mf.Gradients().optimizer(solver='geomeTRIC').kernel(params)
    return mol_eq

def pdb_cap(u,sel_1,sel_2,cap=cap_H_general,del_list=[],resnames=['A','B'],path_save='dimer.pdb',MDA_selection='all',cap_list=[[],[]]):
    """
    Generates PDB file for isolated molecule from MD trajectory. Follows similar format to Process_MD function.
    Parameters
    ----------
    resnames   : Names given to dimer residues that will be printed in the pdb file
    path_save  : Name and location of pdb file
    MDA_selection='all': Selection in MDAnalysis format to be generated in the pdb file 
                         'all' will include all isolated residues. 
    Returns MDAnalysis object
    """

    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str
    agA = u.select_atoms(make_sel(sel_1))
    agB = u.select_atoms(make_sel(sel_2))

    CofMA = agA.center_of_mass()
    CofMB = agB.center_of_mass()

    namesA_cap, xyzA_cap, typesA = cap(u,agA,sel_1,del_list,cap_list[0])
    namesB_cap, xyzB_cap, typesB  = cap(u,agB,sel_2,del_list,cap_list[1])

    orig_mol = get_mol([xyzA_cap, xyzB_cap], [namesA_cap, namesB_cap], [typesA, typesB], 
                       res_names=resnames)

    #save pdb
    orig_all = orig_mol.select_atoms(MDA_selection)
    orig_all.write(path_save)    
    return orig_all


def Process_MD(u,sel_1,sel_2,cap=cap_H_general,del_list=[],cap_list=[[],[]]):
    """
    Extract coordinates from trajectory (saved in an MDAnalysis universe object)    

    Parameters
    ----------
    u   : MDAnalysis universe
        Object containing MD trajectory
    sel_1 : list
        list of strings with the residue ids of Molecule A.
    sel_2 : list
        list of strings with the residue ids of Molecule B.
    cap: Function
        The function to be used to cap the linkers with H
    Returns
    -------
    coordA,coordB,Rab.

    """
    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str
    agA = u.select_atoms(make_sel(sel_1))
    agB = u.select_atoms(make_sel(sel_2))

    CofMA = agA.center_of_mass()
    CofMB = agB.center_of_mass()    
    
    def coord_save(xyz,names):
        #Convert to pyscf format
        atoms = []
        for i in range(len(xyz)):
            new_atom = [names[i][0],tuple(xyz[i])]
            atoms.append(new_atom)
            
        return atoms

    namesA_cap, xyzA_cap, _ = cap(u,agA,sel_1,del_list,cap_list[0])
    namesB_cap, xyzB_cap, _ = cap(u,agB,sel_2,del_list,cap_list[1])

    coordA = coord_save(xyzA_cap,namesA_cap)
    coordB = coord_save(xyzB_cap,namesB_cap)
    
    Rab = CofMA-CofMB    
    return coordA,coordB,Rab

def get_mol(coords, names, types, res_names):
    """ 
    Creates new MDAnalysis object based on given paramenters

    Parameters
    ----------
    coords : list
        List of NumPy arrays/lists of length Nmolecules with atomic coordinates.
    names : list
        List of NumPy arrays/lists of length Nmolecules with atomic names.
    types : list
        List of NumPy arrays/lists of length Nmolecules with atomic types.
    res_names : list
        List of strings with residue names.

    Returns
    -------
    mol_new : MDAnalisys.AtomGroup
        Transformed molecule object.

    """
    if not len(coords) == len(names) == len(types) == len(res_names):
        raise ValueError("All input arrays must be of length Nmolecules")

    n_residues = len(res_names)
    #Creating new molecules
    resids = []
    natoms = 0
    for imol in range(n_residues):
        natom = len(names[imol])
        resid = [imol]*natom
        resids.append(resid)
        natoms += natom
        
    resids = np.concatenate(tuple(resids))
    assert len(resids) == natoms
    segindices = [0] * n_residues
    
    atnames = np.concatenate(tuple(names))
    attypes = np.concatenate(tuple(types))
    resnames = res_names

    mol_new = MDAnalysis.Universe.empty(natoms, 
                                        n_residues=n_residues, 
                                        atom_resindex=resids,
                                        residue_segindex=segindices,
                                        trajectory=True)
    
    mol_new.add_TopologyAttr('name', atnames)
    mol_new.add_TopologyAttr('type', attypes)
    mol_new.add_TopologyAttr('resname', resnames)
    mol_new.add_TopologyAttr('resid', list(range(1, n_residues+1)))
    mol_new.add_TopologyAttr('segid', ['MOL'])  
    mol_new.add_TopologyAttr('id', list(range(natoms)))
    
    # Adding positions
    coord_array = np.concatenate(tuple(coords))
    assert coord_array.shape == (natoms, 3)
    mol_new.atoms.positions = coord_array

    return mol_new
