#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:30:47 2020

@author: mariacm
"""
import os 
os.chdir('/Volumes/Extreme_SSD/PhD_project/Code backup/Squarine_project/')

import numpy as np
from numpy import linalg as LA
from math import sqrt, pi, cos, sin

#MD analysis tools
import MDAnalysis as mda
import couplingutils as cu



ams_au = 0.52917721092
cm1toHartree = 4.5563e-6

# Rotation operators 
Rz = lambda th: np.array([[cos(th),-sin(th),0],
                          [sin(th),cos(th),0],
                          [0,0,1]])
Ry = lambda th: np.array([[cos(th),0,sin(th)],
                          [0,1,0],
                          [-sin(th),0,cos(th)]])
Rx = lambda th: np.array([[1,0,0],
                          [0,cos(th),-sin(th)],
                          [0,sin(th),cos(th)]])


def get_namelist(param, traj_path, traj_num, resnum="11"):
    
    u = mda.Universe(param, traj_path+str(traj_num)+".nc", format="TRJ")
    
    sel1 = u.select_atoms("resid "+resnum)
    
    return sel1.atoms.names
    
    

def get_dihedral(param, traj_path, ntrajs=8, reps=['A','B'],resnum1='11',resnum2='12'):
    
    both_dih1 = []
    both_dih2 = []    
    

    for rep in reps:

        dih1 = []
        dih2 = []
        for ic in range(1,ntrajs+1):   
         
            ##Importing MD trajectories
            u = mda.Universe(param, traj_path+rep+str(ic)+".nc", format="TRJ")
            #u = mda.Universe("%s/Sq%s_dimer.prmtop"%(path,typ), "%s/prod/Sq%sDim_prod%s"%(path,typ,rep)+str(ic)+".nc", format="TRJ")
    
            tin = 0
            dt = 5
            
            print("Traj #%s, rep %s" %(str(ic),rep))
            
            num_sol = 700
            extra_t = 0
            istep = 1
    
            for ts in u.trajectory[num_sol*(istep-1)+tin+extra_t:num_sol*istep+tin+extra_t:dt]:
                            
                sel1 = u.select_atoms("resid "+resnum1)
                sel2 = u.select_atoms("resid "+resnum2)
                
                dihedrals1 = sel1.dihedrals
                dihedrals2 = sel2.dihedrals
     
                dihs1 = dihedrals1.dihedrals()
                dihs2 = dihedrals2.dihedrals()
            
                dih1.append(dihs1)
                dih2.append(dihs2)
                
        dihs1 = np.array(dih1)
        dihs2 = np.array(dih2)
    
        both_dih1.append(dih1)
        both_dih2.append(dih2)
        
    dih_1 = np.average(np.array(both_dih1),axis=0)    
    dih_2 = np.average(np.array(both_dih2),axis=0)  
    
    atom1_1 = dihedrals1.atom1.names
    atom2_1 = dihedrals1.atom2.names
    atom3_1 = dihedrals1.atom3.names
    atom4_1 = dihedrals1.atom4.names
    
    atom1_2 = dihedrals2.atom1.names
    atom2_2 = dihedrals2.atom2.names
    atom3_2 = dihedrals2.atom3.names
    atom4_2 = dihedrals2.atom4.names

    
    return dih_1,dih_2, (atom1_1,atom2_1,atom3_1,atom4_1), (atom1_2,atom2_2,atom3_2,atom4_2)


def RAB_calc(u, dt, selA, selB, Rabs=[]):
    
    num_sol = len(u.trajectory)
    istep = 1
        
    for ts in u.trajectory[num_sol*(istep-1):num_sol*istep:dt]:
                    
        sel1 = u.select_atoms(selA)
        sel2 = u.select_atoms(selB)
        
        CofMA = sel1.center_of_mass()
        CofMB = sel2.center_of_mass()
        
        rab = abs(CofMA-CofMB)
        Rabs.append(rab)    
    
    return Rabs

def get_RAB(param, traj_path, rep, ntrajs=8, traji=1, dt=5, resnum1='11', resnum2='12'):
        
    selA = "resid "+resnum1
    selB = "resid "+resnum2

    Rabs = []
    for ic in range(traji,ntrajs+1):   
     
        ##Importing MD trajectories
        u = mda.Universe(param, traj_path+rep+str(ic)+".nc", format="TRJ")
   
        
        print("RAB Traj #%s, rep %s" %(str(ic),rep))
        
        Rabs = RAB_calc(u, dt, selA, selB, Rabs=Rabs)
            
    Rabs = np.array(Rabs)
    RAB = np.linalg.norm(Rabs,axis=1)
    print(RAB.shape)
     
    # RAB = np.average(np.array(both_RAB),axis=0)
    return RAB



def get_coords(path, param_file, select, file_idx=None, dt=2, resnum1='11', resnum2='12', cap=True):
    """
    Given a selected time in a given traj file, it returns the (MDAnalysis) molecule params.    

    Parameters
    ----------
    path : String
        Location of traj files.
    tdye : String
        "I" or "II", the type of the dimer.
    select : int
        The index of the time-frame to extract.
    rep : String
        "A", "B", "C" or "D"

    Returns
    -------
    Parameters of H-capped Dimer at the selected time-frame.
    Format is: ( xyzA, xyzB, namesA, namesB, atom_typesA, atom_typesB, 
                 [list of bonds]A, [list of bonds]B )

    """
    
    from couplingutils import cap_H
    
    if path[-2:] == 't7':
        #To get universe from a rst7 file
        u = mda.Universe(param_file, path, format="RESTRT")
        select = 0
        dt = 1
    elif path[-2:] == 'db':
        #To get universe from a pdb file
        u = mda.Universe(path, format="PDB")
        select = 0
        dt = 1
    else:
        #To get universe form a nc trajectory
        traj_file = path
        u = mda.Universe(param_file, traj_file+str(file_idx)+".nc", format="TRJ")
        print("The param file is: %s \n" % (param_file),
              "And the traj files is: ",traj_file+str(file_idx)+".nc")
    

    for fi, ts in enumerate(u.trajectory[::dt]): 
            
        if fi == select:
            
            agA = u.select_atoms("resid "+str(resnum1))
            agB = u.select_atoms("resid "+str(resnum2))      
            
            # Getting all parameters from MDAnalysis object
            xyzA = agA.positions
            xyzB = agB.positions
            namesA = agA.atoms.names
            namesB = agB.atoms.names 
            typA = agA.atoms.types
            typB = agB.atoms.types
            
            # First 4 bonds aren't accurate
            bondsA = agA.atoms.bonds.to_indices()[4:]
            bondsB = agB.atoms.bonds.to_indices()[4:]
            idsA = agA.atoms.ids
            idsB = agB.atoms.ids
            
            if cap:
                namesA, xyzA, typA, bondsA = cap_H(u,xyzA,namesA,typA,bondsA,idsA,resnum1)
                namesB, xyzB, typB, bondsB = cap_H(u,xyzB,namesB,typB,bondsB,idsB,resnum2)
            
            CofMA = agA.center_of_mass()
            CofMB = agB.center_of_mass()

    return xyzA, xyzB, namesA, namesB, typA, typB, bondsA, bondsB, CofMA, CofMB

def get_pyscf(traj_file, param, select, resnums, new_coords=None):
    """
    Given a selected time-frame, it returns the molecule's coords as a PySCF 
        formatted string.

    Parameters
    ----------
    path : String
        Location of traj files.
    tdye : String
        "I" or "II", the type of the dimer.
    select : int
        The index of the time-frame to extract.
    rep : String
        "A", "B", "C" or "D"
    new_coords : numpy array, optional
       If given, modified coordinates are used instead of those in the frame.

    Returns
    -------
    xyzA, xyzB : numpy arrays with coordinates in PySCF format

    """

    dt = 20
    time_i = select * dt

    if select < 100:

        i_file = int(time_i/250) + 1
        idx = select*2 % 25
        dt = 5
    else:

        i_file = int((time_i-2000)/1000) + 1
        idx = (select-100) % 100
        dt = 2

    u = mda.Universe(param, traj_file+str(i_file)+".nc", format="TRJ")
    for fi, ts in enumerate(u.trajectory[::dt]):

        if fi == idx:

            sel1, sel2 = resnums

            xyzA,xyzB,RAB = cu.Process_MD(u,sel1,sel2,coord_path="coord_files/MD_atoms", new_coords=new_coords)

    return xyzA, xyzB

def coord_transform(xyzA, xyzB, namesA, namesB, rot_angles, dr=None, assign=None):
    """ Transform given coordinates by rotation and translation. 
            Can be used for any number of molecules of two types, A and B.

    Parameters
    ----------
    xyzA, xyzB : Numpy array or list
        (natoms, 3) array with atoms positions (for A and B-type monomers).
    namesA, namesB : Numpy array or list
        (natoms,) array with atoms names (for A and B-type monomers).

    rot_angles : list of tuples
        [(x1,y1,z1), (x2,y2,z2), ..., (xN,yN,zN)] list of rotation angles
    dr : list, optional
        [[x1,y1,z1], ..., [x1,y1,z1]]. If given, indicates translation displacement
    assign : list, optional
        If given, indicates ordering ot atom types, eg., default [0,1,0,1...] 
         corresponds to N molecules with alternating types.
        
 
    Returns
    -------
    xyz_all : List of len Nmolecules with the atomic coords of each.
    atA, atB : List of atomic indexes for each molecule-type.
    atnonH_A, atnonH_B : List of indexes for non-H atoms in each molecule type.

    """
    nmols = len(rot_angles)
    if dr is None:
        dr = [0,0,0]*nmols
    if assign is None:
        assign = ([0,1]*nmols)[:nmols]
    
    natomsA, natomsB = len(namesA), len(namesB)
    atA = np.arange(natomsA)
    atB = np.arange(natomsB)
    
    #list of non-H atoms
    nonHsA = np.invert(np.char.startswith(namesA.astype(str),'H'))
    atnonH_A = atA[nonHsA]

    nonHsB = np.invert(np.char.startswith(namesB.astype(str),'H'))
    atnonH_B = atB[nonHsB]
    
    mol_list = np.array([xyzA,xyzB]*len(assign))
    coord_list = mol_list[assign]
    
    # Loop over each molecules
    xyz_all = []
    for imol in range(nmols):
        
        xyz0 = coord_list[imol]
        
        # translate to desired position
        xyz = xyz0 + dr[imol]
        
        # rotate
        rx, ry, rz = rot_angles[imol]
        xyz_i = np.dot(Rz(rz),np.dot(Ry(ry),np.dot(Rx(rx),xyz.T))).T
        
        xyz_all.append(xyz_i)
                
    return xyz_all, atA, atB, atnonH_A, atnonH_B

def atom_dist(a1,a2,coord1,coord2):
    """
    Distance between two atoms

    Parameters
    ----------
    a1 : int
        index of atom 1.
    a2 : int
        index of atom 2.
    coord1 : ndarray
        array listing molecule's #1 coordinaties.
    coord2 : TYPE
        array listing molecule's #2 coordinaties.

    Returns
    -------
    dist : float
        in Amstrongs
    """
    dist =LA.norm(coord2[a2] - coord1[a1])
    return dist

def multipole_coup(pos1, pos2, ch1, ch2, at1, at2, atH1, atH2):
    """
    Calculates multiple coupling from inter-atomic distance and atomic excited
        state partial charges

    Parameters
    ----------
    pos1, pos2 : ndarray
        cartesian coord of atoms in molecule 1 and 2.
    ch1, ch2 : ndarray
        array with Lowdin partial charges from tddft for molecules 1 and 2.
    at1, at2 : ndarray
        list of indexes of atoms in molecule 1 and 2.

    Returns
    -------
    Vij : (float) mulitipole coupling

    """   
    from scipy.spatial.distance import cdist
    
    atH1 -= at1[0]
    atH2 -= at2[0]
    
    at1 -= at1[0]
    at2 -= at2[0]
    
    #distsum = np.array([[atom_dist(a1,a2,pos1,pos2) for a2 in at2-1] for a1 in at1-1]) #array (natoms1 x natoms2)
    distsum = cdist(pos1,pos2) 
    
    # Distance between non-H atoms
    #distsum_noH = np.array([[atom_dist(a1,a2,pos1,pos2) for a2 in atH2-1] for a1 in atH1-1]) #array (natoms1 x natoms2)
    pos_nH1 = pos1[atH1]
    pos_nH2 = pos2[atH2]
    distsum_noH = cdist(pos_nH1, pos_nH2)


    # Test if atoms are too close together (approx fails)
    if np.count_nonzero((np.abs(distsum_noH) < 2.0)) > 5 or np.any((np.abs(distsum_noH) < 1.0)):
        #print('tot = ', distsum[np.abs(distsum) < 2.0],', from:', len(at1),len(at2))
        Vij = 9999999999999999
        # print(np.count_nonzero((np.abs(distsum_noH) < 2.0)))
    else:
        Vij = np.sum( np.outer(ch1,ch2)/distsum ) 
        #np.sum( np.multiply(np.outer(ch1,ch2),1/distsum) ) #SUM_{f,g}[ (qf qg)/|rf-rg| ]
        #print('!!',np.count_nonzero((np.abs(distsum_noH) < 2.0)), 'from:', len(at1),len(at2))

    return Vij

def get_mol(coords, names, types, bonds, res_names):
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
    bondsA : list
        List of NumPy arrays/lists of length Nmolecules with atomic bonds.
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
        
    resids = np.concatenate(tuple(resids))#np.concatenate((np.full((nat1,1), 0),np.full((nat2,1), 1)),axis=0).T
    assert len(resids) == natoms
    segindices = [0] * n_residues
    
    atnames = np.concatenate(tuple(names))
    attypes = np.concatenate(tuple(types))
    resnames = res_names

    mol_new = mda.Universe.empty(natoms, 
                                 n_residues=n_residues, 
                                 atom_resindex=resids,
                                 residue_segindex=segindices,
                                 trajectory=True)
    
    mol_new.add_TopologyAttr('name', atnames)
    mol_new.add_TopologyAttr('type', attypes)
    mol_new.add_TopologyAttr('resname', resnames)
    mol_new.add_TopologyAttr('resid', list(range(1, n_residues+1)))
    mol_new.add_TopologyAttr('segid', ['SQA'])  
    mol_new.add_TopologyAttr('id', list(range(natoms)))
    
    # Adding positions
    coord_array = np.concatenate(tuple(coords))
    assert coord_array.shape == (natoms, 3)
    mol_new.atoms.positions = coord_array

    # Adding bonds
    """
    n_acum = 0
    all_bonds = []
   
    for imol in range(n_residues):
        bond = bonds[imol] - np.min(bonds[imol])
        d_bd = [bond[i] + n_acum for i in range(len(bond))]
        all_bonds.append(d_bd)
        n_acum += len(names[imol])
    """
     
    #bonds0 = np.concatenate(tuple(all_bonds),axis=0)
    #atbonds = list(map(tuple, bonds0))

    #mol_new.add_TopologyAttr('bonds', atbonds)
    
    return mol_new

def get_charges(u, pos_ion, neg_ion):
    """
    Given a trajectory frame, it returns the coordinates of CofM
    for all charged elements (DNA residues and ions) in the environment.
    This function is to be called inside a trajectory loop.

    Parameters
    ----------
    u : MDAnalysis universe
        Traj frame
    pos_ion: Tuple (str, int)
        (resname, charge) of the positive solvent ion 
    neg_ion: Tuple (str, int)
        (resname, charge) of the negative solvent ion 

    Returns
    -------
    Coordinates of charged elements as (coord, charge) tuples: 
    (DNA, pos ions, neg ions)

    """ 
    pos_name, pos_charge = pos_ion
    neg_name, neg_charge = neg_ion

    dna_res = u.select_atoms("nucleic")
    dna_coords = []
    dna_charges = np.array([-1.0] * dna_res.residues.n_residues)
    for res in dna_res.residues:
        cofm = res.atoms.center_of_geometry()
        dna_coords.append(cofm)

    pos_res = u.select_atoms("resname " + pos_name)
    pos_coords = []
    pos_charges = np.array([pos_charge] * pos_res.residues.n_residues)
    for res in pos_res.residues:
        cofm = res.atoms.center_of_geometry()
        pos_coords.append(cofm)

    neg_res = u.select_atoms("resname " + neg_name)
    neg_coords= []
    neg_charges = np.array([neg_charge] * neg_res.residues.n_residues)
    for res in neg_res.residues:
        cofm = res.atoms.center_of_geometry()
        neg_coords.append(cofm)
    
    coords = np.concatenate((dna_coords, pos_coords, neg_coords),axis=0)
    charges = np.concatenate((dna_charges, pos_charges, neg_charges),axis=0)
    return coords, charges


def solvent_coords(path, tdye, select, rep):
    """
    Given a selected time-frame, it returns the (MDAnalysis) molecule params.    

    Parameters
    ----------
    path : String
        Location of traj files.
    tdye : String
        "I" or "II", the type of the dimer.
    select : int
        The index of the time-frame to extract.
    rep : String
        "A", "B", "C" or "D"

    Returns
    -------
    Parameters of H-capped Dimer at the selected time-frame.
    Format is: ( xyzA, xyzB, namesA, namesB, atom_typesA, atom_typesB, 
                 [list of bonds]A, [list of bonds]B )

    """
    
    typ = 'Opp' if tdye == 'II' else ''
    if rep == 'C':
        param_files = lambda ty: 'Sq' + ty + '_dimer_g1.prmtop'
    elif rep == 'D':
        param_files = lambda ty: 'Sq' + ty + '_dimer_g2.prmtop'
    else:
        param_files = lambda ty: 'Sq' + ty + '_dimer.prmtop'

    dt_both = 20
    time_i = select * dt_both
    
    if select < 100:
        i_file = int(time_i/250) + 1 
        idx = select*2 % 25
        dt = 5
    else:
        last_file = 8
        i_file = int((time_i-2000)/1000) + 1 + last_file
        idx = (select-100) % 100
        dt = 2

    
    u = mda.Universe("%s/%s" % (path, param_files(typ)),
                            "%s/prod/Sq%sDim_prod%s" % (path, typ, rep) + str(i_file) + ".nc", format="TRJ")
    print("The param file is: %s/%s \n" % (path, param_files(typ)),
          "And the traj files is: %s/prod/Sq%sDim_prod%s" % (path, typ, rep) + str(i_file) + ".nc")
    for fi, ts in enumerate(u.trajectory[::dt]): 
            
        if fi == idx:
            t_i = round((ts.frame*10),2)
            print("The time-step is: ", t_i)
            
            coord, charge = get_charges(u, ("MG", 2.0), ("Cl-",-1.0))

    return coord, charge

def get_pdb(traj_path, param_path, path_save, resnums, select=(1,0), dt=2, MDA_selection='all'):
    
    i_file, idx = select
    xyza, xyzb, namesA, namesB, typeA, typeB, bondsA, bondsB, __, __ = get_coords(traj_path, param_path, 
                                                                          idx, file_idx=i_file, dt=dt, 
                                                                          resnum1=resnums[0], resnum2=resnums[1])

    orig_mol = get_mol([xyza, xyzb], [namesA, namesB], [typeA, typeB], 
                       [bondsA[:-5], bondsB[:-5]], res_names=['SQA', 'SQB'])

    #save pdb
    orig_all = orig_mol.select_atoms(MDA_selection)
    orig_all.write(path_save)    
    return orig_all
    

def max_pdb(Vi,traj_path,param_path,resnums,path_save,sel='all'):
    """
    Generate a pdb of the dimer coordinates at max VFE/VCT

    Returns
    -------
    None.

    """
    
    max_V = np.argmax(Vi)
    print(max_V)
    
    dt_both = 20
    time_i = max_V * dt_both
    
    '''
    if max_V < 100:

        i_file = int(time_i/250) + 1
        idx = max_V*2 % 25
        dt = 5
    
    else:
    '''
    last_file = 6
    i_file = int((time_i-2000)/1000) + 1 + last_file
    idx = (max_V-100) % 100
    dt = 1
    
    
    obj = get_pdb(traj_path, param_path, path_save, resnums, select=(i_file,idx), 
            dt=dt, MDA_selection=sel)
    
    
    return obj

def COM_atom(COM, xyz, names):
    diff = abs(xyz - COM)
    diff_glob = np.mean(diff, axis=1)
    closer_idx = np.argmin(diff_glob)
    print(diff.shape)
    
    # We are not interested in H or O
    while names[closer_idx][0] == 'H' or names[closer_idx][0] == 'O':
        print(names[closer_idx])
        diff = np.delete(diff, closer_idx, axis=0)
        diff_glob = np.mean(diff, axis=1)
        closer_idx = np.argmin(diff_glob, axis=0)
        
    
    return closer_idx, xyz[closer_idx], names[closer_idx]
    

def energy_diagram(file_list, global_min, time):
    eV_conv = 27.211399
    energies = np.empty((0,2))
    for f in file_list:
        data = np.loadtxt(f)
        energies = np.concatenate((energies,data))

    energies = eV_conv*(energies-global_min)
        
    return energies
            
def displaced_dimer(sel1, sel2, cof_dist, disp, 
                    atom_orig='N1', atom_z='N2', atom_y='C2', 
                    res_names=['SQA', 'SQB']):
    
    
    xyz1 = sel1.positions 
    xyz2 = sel2.positions
    
    
    idx1A = np.nonzero(sel1.atoms.names==atom_orig)[0][0]
    idx1B = np.nonzero(sel1.atoms.names==atom_z)[0][0]
    idx1C = np.nonzero(sel1.atoms.names==atom_y)[0][0]
    idx2A = np.nonzero(sel2.atoms.names==atom_orig)[0][0]
    idx2B = np.nonzero(sel2.atoms.names==atom_z)[0][0] 
    idx2C = np.nonzero(sel2.atoms.names==atom_y)[0][0] 

    x_unit = np.array([1,0,0]).reshape(1,-1)   
    y_unit = np.array([0,1,0]).reshape(1,-1)
    z_unit = np.array([0,0,1]).reshape(1,-1)
    
    #Rodrigues formula
    def rodrigues(xyz, phi, v_unit):
    
        xyz_new = ((xyz.T*np.cos(phi)).T + np.cross(v_unit[0,:],xyz.T,axis=0).T * np.sin(phi)
                    + (v_unit * np.dot(v_unit,xyz.T).reshape(-1,1)) * (1-np.cos(phi)))    
        
        return xyz_new

    
    #placing atomA of both molecules in the origin
    xyz1 -= xyz1[idx1A]
    xyz2 -= xyz2[idx2A]
    
    #rotating molecule to yz plane
    rot1 = xyz1[idx1B]
    #phiz1 = - np.arccos(rot1[2]/LA.norm(rot1))
    num = rot1[1]*np.sqrt(rot1[0]**2+rot1[1]**2) + rot1[2]**2
    theta1 = -np.arccos(num/LA.norm(rot1)**2)
    xyz1_new = rodrigues(xyz1, theta1, z_unit)

    #rotating to z axis
    rot1 = xyz1_new[idx1B]
    phi1 = np.arccos(rot1[2]/LA.norm(rot1))
    xyz1_new = rodrigues(xyz1_new, phi1, y_unit)
    
    #Rotating the other atom axis
    rot1 = xyz1_new[idx1C]
    psi1 = np.arccos(rot1[1]/LA.norm(rot1))
    xyz1_new = rodrigues(xyz1_new, psi1, x_unit)
    
    ###
    rot2 = xyz2[idx2B]
    num = rot2[1]*np.sqrt(rot2[0]**2+rot2[1]**2) + rot2[2]**2
    theta2 = -np.arccos(num/LA.norm(rot2)**2)
    xyz2_new = rodrigues(xyz2, theta2, z_unit)
    
    #rotating to z axis
    rot2 = xyz2_new[idx2B]
    phi2 = np.arccos(rot2[2]/LA.norm(rot2))
    xyz2_new = rodrigues(xyz2_new, phi2, y_unit)
    
    #Rotating the other atom axis
    rot2 = xyz2_new[idx2C]
    psi2 = np.arccos(rot2[1]/LA.norm(rot2))
    xyz2_new = rodrigues(xyz2_new, psi2, x_unit)
    
    
    #xyz2_new -= xyz2_new[idx2A]
    
    
    #displacing sel2 on the x axis only
    xyz2_new[:,0] += cof_dist
    
    #displacing sel2 on y axis
    xyz2_new[:,2] += disp
    
    #create new MDA object
    mol = get_mol([xyz1_new,xyz2_new], [sel1.atoms.names,sel2.atoms.names], 
                   [sel1.atoms.types,sel2.atoms.types], [sel1.atoms.bonds,sel2.atoms.bonds], 
                   res_names=res_names)
    
    #print(rot1[2]*LA.norm(rot1))
    
    return mol

    
    
    
        
        
        
    