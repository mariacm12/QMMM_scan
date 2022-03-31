#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions for geometry manipulation
@author: mariacm
"""
import os 
os.chdir('/Volumes/Extreme_SSD/PhD_project/Code backup/Squarine_project/')

import numpy as np
from numpy import linalg as LA
from math import sqrt, pi, cos, sin

#MD analysis tools
import MDAnalysis as mda



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

def get_dihedral(param, traj_file, ntrajs=8, reps=['A','B'],resnum1='11',resnum2='12'):
    """
    Returns the list of dihedral angles for a trajectory, averaged over the repetitions.  

    Parameters
    ----------
    param : string. The .prmtop parameter file       
    traj_file : string. The location of the .nc trajectory file, 
                including the file prefix and excluding the rep and traj number identifier.
    ntrajs : int. Total number of trajectory files per rep. The default is 8.
    reps : list. A list of strings with the reps labels (as in the file format).
            The default is ['A','B'].
    resnum1 : string, The residue id of the first monomer. The default is '11'.
    resnum2 : string, The residue id of the second monomer. The default is '12'.

    Returns
    -------
    dih_1/dih_2 : numpy array. Dihedrals for molecules 1 and 2 
    names1. names2 : tuple. Of arrays with the atom names in the dihedral.
                        (atom1, atom2, atom3, atom4)
    

    """
    
    both_dih1 = []
    both_dih2 = []    
    

    for rep in reps:

        dih1 = []
        dih2 = []
        for ic in range(1,ntrajs+1):   
         
            ##Importing MD trajectories
            u = mda.Universe(param, traj_file+rep+str(ic)+".nc", format="TRJ")
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

                dih1.append(dihedrals1.dihedrals())
                dih2.append(dihedrals2.dihedrals())
                
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
    
    names1 = (atom1_1,atom2_1,atom3_1,atom4_1)
    names2 = (atom1_2,atom2_2,atom3_2,atom4_2)

    
    return dih_1, dih_2, names1, names2


def RAB_calc(u, dt, selA, selB, Rabs=[]):
    """
    Function to calculate center of mass distance between A and B.

    Parameters
    ----------
    u : MDAnalysis universe. Object containing loaded trajectory.
    dt : int. Time step. =
    selA/selB : string. Selection keyword for molecules A, B.
    Rabs : list, optional. List of previously calculated distances, if any.

    Returns
    -------
    Rabs : list. Calculated distances. 

    """
    
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

def get_RAB(param, traj_file, ntrajs=8, traji=1, dt=5, resnum1='1', resnum2='2'):
    """
    Returns an array with the center of mass distance along a trajectory. 
    Written QM trajectories splitted in multiple files.

    Parameters
    ----------
    param : string. Paramenter file.
    traj_path : string. Trajectory file prefix (Paht+name excluding traj index and .nc)
    ntrajs : int. Number of trajectory files to include.
    traji : initial trajectory file to include in the output.
    dt : int. Time step.
    resnum1 : string, The residue id of the first monomer. The default is '1'.
    resnum2 : string, The residue id of the second monomer. The default is '2'.

    Returns
    -------
    RAB : NumPy array.Center of mass distance along the trajectory.

    """
        
    selA = "resid "+resnum1
    selB = "resid "+resnum2

    Rabs = []
    for ic in range(traji,ntrajs+1):   
     
        ##Importing MD trajectories
        u = mda.Universe(param, traj_file+str(ic)+".nc", format="TRJ")
   
        
        print("RAB Traj #%s" %(str(ic)))
        
        Rabs = RAB_calc(u, dt, selA, selB, Rabs=Rabs)
            
    Rabs = np.array(Rabs)
    RAB = np.linalg.norm(Rabs,axis=1)
     
    return RAB



def get_coords(traj_file, param_file, select, file_idx=None, dt=2, resnum1='1', resnum2='2', cap=True, del_list=[]):
    """
    Given a selected time in a given traj file, it returns the (MDAnalysis) molecule params.    

    Parameters
    ----------
    traj_file : String. Path of the trajectory file .nc, excluding traj index.
                Also takes full file name of pdb and restrt formats.
    param_file : String. Parameter .prmtop file.
    file_idx : int. The index of the file trajectory to extract.
    select : Index of the frame to extract.
    dt : int. Time step.
    resnum1 : string, The residue id of the first monomer. The default is '1'.
    resnum2 : string, The residue id of the second monomer. The default is '2'.
    cap : Whether to cap phosphate molecules with hydrogen atoms.
    

    Returns
    -------
    Parameters of H-capped Dimer at the selected time-frame.
    Format is: ( xyzA, xyzB, namesA, namesB, atom_typesA, atom_typesB, 
                 [list of bonds]A, [list of bonds]B, CofMA, CofMB)

    """
    
    from couplingutils import cap_H_general
    
    if traj_file[-2:] == 't7':
        #To get universe from a rst7 file
        u = mda.Universe(param_file, traj_file, format="RESTRT")
        select = 0
        dt = 1
    elif traj_file[-2:] == 'db':
        #To get universe from a pdb file
        u = mda.Universe(traj_file, format="PDB")
        select = 0
        dt = 1
    else:
        #To get universe form a nc trajectory
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
            
            if cap: #cap bonds not implemented yet
                namesA, xyzA, typA, [] = cap_H_general(u,agA,resnum1,del_list=del_list)
                namesB, xyzB, typB, [] = cap_H_general(u,agB,resnum2,del_list=del_list)
            
            CofMA = agA.center_of_mass()
            CofMB = agB.center_of_mass()

    return xyzA, xyzB, namesA, namesB, typA, typB, bondsA, bondsB, CofMA, CofMB

def coord_transform(xyzA, xyzB, namesA, namesB, rot_angles, dr=None, assign=None):
    """ 
    Transform given coordinates by rotation and translation. 
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
    a1 : int. Index of atom 1.
    a2 : int. Index of atom 2.
    coord1 : ndarray. Array listing molecule's 1 and 2 coordinates.

    Returns
    -------
    dist : float
        in Amstrongs
    """
    dist =LA.norm(coord2[a2] - coord1[a1])
    return dist

def multipole_coup(pos1, pos2, ch1, ch2, at1, at2, atH1, atH2):
    """
    Calculates Transition Monopole coupling from inter-atomic distance and atomic excited
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
    Vij : float. Coupling according to transition monopoles approximation.

    """   
    from scipy.spatial.distance import cdist
    
    atH1 -= at1[0]
    atH2 -= at2[0]
    
    at1 -= at1[0]
    at2 -= at2[0]
    
    distsum = cdist(pos1,pos2) 
    
    # Distance between non-H atoms
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


def solvent_coords(path, param_file, file_idx, select, dt=2):
    """
    Returns the corrdinates of the solvent charge particles.    

    Parameters
    ----------
    path : String. Path of the trajectory file .nc including file preffix.
    param_file : String. Parameter .prmtop file.
    file_idx : int. The index of the file trajectory to extract.
    select : Index of the frame to extract.
    dt : int. Time step.

    Returns
    -------
    Parameters of H-capped Dimer at the selected time-frame.
    Format is: ( xyzA, xyzB, namesA, namesB, atom_typesA, atom_typesB, 
                 [list of bonds]A, [list of bonds]B )

    """

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
            t_i = round((ts.frame*10),2)
            print("The time-step is: ", t_i)
            
            coord, charge = get_charges(u, ("MG", 2.0), ("Cl-",-1.0))

    return coord, charge

def get_pdb(traj_file, param_file, path_save, resnums, select=(1,0), dt=2, MDA_selection='all'):
    """
    Returns pdb file of trajectory snapshot of indicated residues.

    Parameters
    ----------
    traj_file : String. Path of the trajectory file .nc including file preffix.
    param_file : String. Parameter .prmtop file.
    path_save : String. Path to save pdb file.
    resnums : tuple/list. Residue ids for the dimer molecules.
    select : tuple/list. (File index, frame index).
    dt : int. Time step.
    MDA_selection : string, optional. MDAnalisys selection for the pfb to save.

    Returns
    -------
    orig_all : MDAnalysis object that was saved in the pdb.

    """
    
    i_file, idx = select
    xyza, xyzb, namesA, namesB, typeA, typeB, __, __, __, __ = get_coords(traj_file, param_file, 
                                                                          idx, file_idx=i_file, dt=dt, 
                                                                          resnum1=resnums[0], resnum2=resnums[1])

    orig_mol = get_mol([xyza, xyzb], [namesA, namesB], [typeA, typeB], 
                       res_names=['MOA', 'MOB'])

    #save pdb
    orig_all = orig_mol.select_atoms(MDA_selection)
    orig_all.write(path_save)    
    return orig_all

def COM_atom(COM, xyz, names):
    """
    Find the atom closest to the center of mass in a molecule.

    Parameters
    ----------
    COM : Numpy array. Center of mass (x,y,z) coordinates
    xyz : Numpy array. Atom coordinates.
    names : Numpy array. List of atom names in molecule.

    Returns
    -------
    Index of closest atom.
    (x,y,z) Coordinates of closest atom.
    Name of closes atom

    """
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
    
            
def displaced_dimer(sel1, sel2, cof_dist, disp, 
                    atom_orig='N1', atom_z='N2', atom_y='C2', 
                    res_names=['MOA', 'MOB']):
    """
    Returns dimer where monomers are displaced from each other by the given parameters.

    Parameters
    ----------
    sel1/sel2 : MDAnalsys selection. Molecules 1 and 2.
    cof_dist : Numpy array. Desired center of mass distance between monomers.
    disp : Numpy array. Vertical displacement between molecules.
    atom_orig : string. Atom to be placed at the origin.
    atom_z : string. Atom that will define the z axis vector of the molecule (from the origin).
    atom_y : string. Atom that will define the y axis vector of the molecule.
    res_names : list. Residue names for the final MDAnalysis object. The default is ['SQA', 'SQB'].

    Returns
    -------
    MDAnalysis object of the new dimer.

    """
    

    
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

    
    
    
        
        
        
    
