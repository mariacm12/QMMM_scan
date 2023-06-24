#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:30:47 2020
@author: mariacm
"""
import os

import numpy as np
from numpy import linalg as LA
from math import sqrt, pi, cos, sin

# MD analysis tools
import MDAnalysis as mda
import dyeScreen.commons.couplingutils as cu

import csv

ams_au = 0.52917721092
cm1toHartree = 4.5563e-6

# Rotation operators


def Rz(th): return np.array([[cos(th), -sin(th), 0],
                             [sin(th), cos(th), 0],
                             [0, 0, 1]])


def Ry(th): return np.array([[cos(th), 0, sin(th)],
                             [0, 1, 0],
                             [-sin(th), 0, cos(th)]])


def Rx(th): return np.array([[1, 0, 0],
                             [0, cos(th), -sin(th)],
                             [0, sin(th), cos(th)]])


def get_namelist(param, traj_path, traj_num, resnum="11"):

    u = mda.Universe(param, traj_path+str(traj_num)+".nc", format="TRJ")

    sel1 = u.select_atoms("resid "+resnum)

    return sel1.atoms.names


def get_dihedral(param, traj_path, ntrajs=8, reps=['A', 'B'], resnum1='11', resnum2='12'):

    both_dih1 = []
    both_dih2 = []

    for rep in reps:

        dih1 = []
        dih2 = []
        for ic in range(1, ntrajs+1):

            # Importing MD trajectories
            u = mda.Universe(param, traj_path+rep+str(ic)+".nc", format="TRJ")
            #u = mda.Universe("%s/Sq%s_dimer.prmtop"%(path,typ), "%s/prod/Sq%sDim_prod%s"%(path,typ,rep)+str(ic)+".nc", format="TRJ")

            tin = 0
            dt = 5

            print("Traj #%s, rep %s" % (str(ic), rep))

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

    dih_1 = np.average(np.array(both_dih1), axis=0)
    dih_2 = np.average(np.array(both_dih2), axis=0)

    atom1_1 = dihedrals1.atom1.names
    atom2_1 = dihedrals1.atom2.names
    atom3_1 = dihedrals1.atom3.names
    atom4_1 = dihedrals1.atom4.names

    atom1_2 = dihedrals2.atom1.names
    atom2_2 = dihedrals2.atom2.names
    atom3_2 = dihedrals2.atom3.names
    atom4_2 = dihedrals2.atom4.names

    return dih_1, dih_2, (atom1_1, atom2_1, atom3_1, atom4_1), (atom1_2, atom2_2, atom3_2, atom4_2)


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
    for ic in range(traji, ntrajs+1):

        # Importing MD trajectories
        u = mda.Universe(param, traj_path+rep+str(ic)+".nc", format="TRJ")

        print("RAB Traj #%s, rep %s" % (str(ic), rep))

        Rabs = RAB_calc(u, dt, selA, selB, Rabs=Rabs)

    Rabs = np.array(Rabs)
    RAB = np.linalg.norm(Rabs, axis=1)
    print(RAB.shape)

    # RAB = np.average(np.array(both_RAB),axis=0)
    return RAB


def get_coords_old(path, param_file, select, file_idx=None, dt=2, resnum1='11', resnum2='12', cap=True):
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


    if path[-2:] == 't7':
        # To get universe from a rst7 file
        u = mda.Universe(param_file, path, format="RESTRT")
        select = 0
        dt = 1
    elif path[-2:] == 'db':
        # To get universe from a pdb file
        u = mda.Universe(path, format="PDB")
        select = 0
        dt = 1
    else:
        # To get universe form a nc trajectory
        traj_file = path
        u = mda.Universe(param_file, traj_file +
                         str(file_idx)+".nc", format="TRJ")
        print("The param file is: %s \n" % (param_file),
              "And the traj files is: ", traj_file+str(file_idx)+".nc")

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
                namesA, xyzA, typA, bondsA = cu.cap_H(
                    u, xyzA, namesA, typA, bondsA, idsA, resnum1)
                namesB, xyzB, typB, bondsB = cu.cap_H(
                    u, xyzB, namesB, typB, bondsB, idsB, resnum2)

            CofMA = agA.center_of_mass()
            CofMB = agB.center_of_mass()

    return xyzA, xyzB, namesA, namesB, typA, typB, bondsA, bondsB, CofMA, CofMB


def get_coords(path, param_file, select, file_idx=None, dt=2, sel_1=['1'], sel_2=None,
               cap=True, del_list=[], cap_list=[[], []], resnames=False):
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
    (xyz, names, atom_types)
    If sel_2!=None, each is a list of size 2: [sthA, sthB]
    """

    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str

    if path[-2:] == 't7':
        # To get universe from a rst7 file
        u = mda.Universe(param_file, path, format="RESTRT")
        select = 0
        dt = 1
    elif path[-2:] == 'db':
        # To get universe from a pdb file
        u = mda.Universe(path, format="PDB")
        select = 0
        dt = 1
    else:
        # To get universe form a nc trajectory
        traj_file = path
        fidx = str(file_idx) if file_idx is not None else ''
        print("The param file is: %s \n" % (param_file),
              "And the traj files is: ", traj_file+fidx+".nc")
        u = mda.Universe(param_file, traj_file+fidx+".nc")

    for fi, ts in enumerate(u.trajectory[::dt]):
        if fi == select:
            # Getting all parameters from MDAnalysis object
            agA = u.select_atoms(make_sel(sel_1))
            xyz = [agA.positions]
            names = [agA.atoms.names]
            typ = [agA.atoms.types]
            CofM = [agA.center_of_mass()]
            resname = ['0']
            if resname:
                resname = [agA.resnames]

            if sel_2 is not None:  # A dimer
                agB = u.select_atoms(make_sel(sel_2))
                xyz += [agB.positions]
                names += [agB.atoms.names]
                typ += [agB.atoms.types]
                CofM += [agB.center_of_mass()]
                if resname:
                    resname += [agB.resnames]

            if cap:
                names, xyz, typ = [cu.cap_H_general(
                    u, agA, sel_1, del_list, cap_list[0])]
                if sel_2 is not None:
                    namesB, xyzB, typB = cu.cap_H_general(
                        u, agB, sel_2, del_list, cap_list[1])
                    names += namesB
                    xyz += xyzB
                    typ += typB
                print(del_list)

    return xyz, names, typ, CofM, resname


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

            xyzA, xyzB, RAB = cu.Process_MD(
                u, sel1, sel2, coord_path="coord_files/MD_atoms", new_coords=new_coords)

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
        dr = [0, 0, 0]*nmols
    if assign is None:
        assign = ([0, 1]*nmols)[:nmols]

    natomsA, natomsB = len(namesA), len(namesB)
    atA = np.arange(natomsA)
    atB = np.arange(natomsB)

    # list of non-H atoms
    nonHsA = np.invert(np.char.startswith(namesA.astype(str), 'H'))
    atnonH_A = atA[nonHsA]

    nonHsB = np.invert(np.char.startswith(namesB.astype(str), 'H'))
    atnonH_B = atB[nonHsB]

    mol_list = np.array([xyzA, xyzB]*len(assign))
    coord_list = mol_list[assign]

    # Loop over each molecules
    xyz_all = []
    for imol in range(nmols):

        xyz0 = coord_list[imol]

        # translate to desired position
        xyz = xyz0 + dr[imol]

        # rotate
        rx, ry, rz = rot_angles[imol]
        xyz_i = np.dot(Rz(rz), np.dot(Ry(ry), np.dot(Rx(rx), xyz.T))).T

        xyz_all.append(xyz_i)

    return xyz_all, atA, atB, atnonH_A, atnonH_B


def coord_transform_single(xyzA, xyzB, namesA, namesB, rot_angles, dr=None, del_ats=None):
    """ Transform given coordinates by rotation and translation. 
        Simplifies version for a single dimer.
    Parameters
    ----------
    xyzA, xyzB : Numpy array or list
        (natoms, 3) array with atoms positions.
    namesA, namesB : Numpy array or list
        (natoms,) array with atoms names.
    rot_angles : list of tuples
        [(x1,y1,z1), (x2,y2,z2)] list of rotation angles
    dr : list, optional
        [[x1,y1,z1], [x2,y2,z2]]. If given, indicates translation displacement
    del_ats : list, optional
        If given indicates the indexes of the atoms to delete from the molecules.
        Must be given as a list [idx_A,idx_B]

    Returns
    -------
    xyzA, xyzB : Atomic coords of A and B molecules.
    atA, atB : List of atomic indexes for A and B molecules.
    atnonH_A, atnonH_B : List of indexes for non-H atoms in each molecule.
    """

    if del_ats is not None:
        namesA = del_atoms(namesA, namesA, del_ats[1])
        namesB = del_atoms(namesB, namesB, del_ats[0])
        xyzA = del_atoms(xyzA, namesA, del_ats[1])
        xyzB = del_atoms(xyzB, namesB, del_ats[0])

    if dr is None:
        dr = [[0, 0, 0], [0, 0, 0]]

    natomsA, natomsB = len(namesA), len(namesB)
    atA = np.arange(natomsA)
    atB = np.arange(natomsB)

    # list of non-H atoms
    nonHsA = np.invert(np.char.startswith(namesA.astype(str), 'H'))
    atnonH_A = atA[nonHsA]

    nonHsB = np.invert(np.char.startswith(namesB.astype(str), 'H'))
    atnonH_B = atB[nonHsB]

    # translate to desired position
    xyzA += dr[0]
    xyzB += dr[1]

    # rotate
    rx1, ry1, rz1 = rot_angles[0]
    rx2, ry2, rz2 = rot_angles[1]
    xyz_A = np.dot(Rz(rz1), np.dot(Ry(ry1), np.dot(Rx(rx1), xyzA.T))).T
    xyz_B = np.dot(Rz(rz2), np.dot(Ry(ry2), np.dot(Rx(rx2), xyzB.T))).T

    return xyz_A, xyz_B, namesA, namesB, atA, atB, atnonH_A, atnonH_B


class scan_cofigs():

    def __init__(self, type_of_attach='single', dna_stericBox=20) -> None:
        if type_of_attach == 'single':
            self.num_attachs = 1
        elif type_of_attach == 'double':
            self.num_attachs = 2
        else:
            raise NotImplementedError(
                "Only single and double attachements to DNA are allowed")

        self.attach_idxs = []

        # How big of a "box" should we consider when calculating steric interactions
        self.dna_sbox = dna_stericBox

        self.pairs = []
        self.bonds = []
        self.del_res = ""
        self.dye_del = " "

    class DNA():
        def __init__(self, dnaU, chain=None) -> None:
            # Select only viable chain
            if chain: 
                self.chain_sel = dnaU.select_atoms("segid " + chain)
            else: 
                self.chain_sel = dnaU.select_atoms("all")
            
            # Properties
            self.atoms = self.chain_sel.atoms
            self.positions = self.atoms.positions
            self.names = self.atoms.names
            self.com = self.atoms.center_of_mass()
            # list of non-H atoms
            self.natoms = len(self.names)
            self.atIds = np.arange(self.natoms)
            nonHs = np.invert(np.char.startswith(self.names.astype(str), 'H'))
            self.heavyIds = self.atIds[nonHs]

        def get_ter(self):
            ter_res = []
            for r in self.chain_sel.residues:
                if "HO3'" in r.atoms.names:
                    ter_res.append(r.ix)
            
            return ter_res
        
        def select_atoms(self, string):
            return self.chain_sel.select_atoms(string)

    def extract_monomer(self, pdb_monomer, resD, resL, attach_points, pre_attach):
        '''Extract coords and info from monomer pdb
        '''

        sel_1 = [resD, resL] if resL else [resD]
        xyz, names, typ, com, reslist = get_coords(pdb_monomer, None, 0, sel_1=sel_1, cap=False, resnames=True)
        self.xyzA = np.array(xyz[0])
        self.xyzB = np.array(xyz[0])

        self.comA = com[0]
        self.comB = com[0]

        self.names = names[0]
        self.types = typ[0]
        self.reslist = reslist

        # rename pre-attachement oxygens for DNA FF name
        if pre_attach:
            self.names[self.names == pre_attach[0]] =  "O3'"
            self.names[self.names == pre_attach[1]] =  "O5'"

        # list of non-H atoms
        self.natoms = len(self.names)
        self.atIds = np.arange(self.natoms)
        nonHs = np.invert(np.char.startswith(self.names.astype(str), 'H'))
        self.heavyIds = self.atIds[nonHs]

        for a in range(self.num_attachs):
            attach_idx = np.where(self.names == attach_points[a])[0][0]
            self.attach_idx = attach_idx
            self.attach_idxs.append(attach_idx)

        return

    def extract_DNA(self, pdb_dna, chain=None):
        ''' Extract info from DNA PDB given the allowed chain segid
        '''
        self.u_DNA = mda.Universe(pdb_dna, format="PDB")
        self.dna = self.DNA(self.u_DNA, chain=chain) 

        # Resids from terminal atoms
        self.res_ter =  self.dna.chain_sel.select_atoms("name HO3'").resids
        # index atom (Choosing atom that's always kept)
        self.res_idx =  self.dna.chain_sel.select_atoms("name C3'").positions

        # Extract positions and info of all P atoms in every nucleotide
        self.bond_pos = self.dna.chain_sel.select_atoms("name P").positions
        self.bond_resn = self.dna.chain_sel.select_atoms("name P").resnames
        self.bond_resi = self.dna.chain_sel.select_atoms("name P").resids
        self.bond_labels = [self.bond_resn[i] +
                            str(self.bond_resi[i]) for i in range(len(self.bond_resn))]

        print("atoms count", self.dna.natoms,
              len(self.dna.chain_sel.atoms.positions))

        return

    def move_single_att(self, a, b):
        ''' Moving coordinate for a specific sampling 
        '''
        # Define DNA box
        dna_stericA = self.get_DNABox(self.dna_sbox, mid=self.bond_pos[a])
        dna_stericB = self.get_DNABox(self.dna_sbox, mid=self.bond_pos[b])

        new_xyzA, newcomA = align_two_molecules_at_one_point(self.xyzA, self.bond_pos[a], self.xyzA[self.attach_idx], 
                                                            dna_stericA.com, self.comA, align=-1, accuracy=0.7)
        new_xyzB, newcomB = align_two_molecules_at_one_point(self.xyzB, self.bond_pos[b], self.xyzB[self.attach_idx],
                                                            dna_stericB.com, self.comB, align=-1, accuracy=0.7)

        '''
        # Move monA to point a, and monB to point b
        tvectorA = self.bond_pos[a] - self.xyzA[self.attach_idx]
        tvectorB = self.bond_pos[b] - self.xyzB[self.attach_idx]
        new_xyzA = self.xyzA + tvectorA
        new_xyzB = self.xyzB + tvectorB
        '''

        # Appending current bond information
        pairs = self.bond_labels[a]+'-'+self.bond_labels[b]
        self.bonds.append([pairs, self.bond_pos[a][0], self.bond_pos[b][0]])

        # Have to "manually" make the list of resnames and resids so it matches the mol2 files
        resname1 = self.reslist[0][0]
        resname2 = self.reslist[0][-1]
        self.reslist_dimer = [resname1, resname2]*2  # 4 total residues
        natoms1, natoms2 = np.count_nonzero(
            self.reslist[0] == resname1), np.count_nonzero(self.reslist[0] == resname2)
        self.resid_list = [0]*natoms1 + [1]*natoms2 + [2]*natoms1 + [3]*natoms2

        self.dimer_mol = get_mol([new_xyzA, new_xyzB], [self.names]*2, [self.types]*2,
                                  res_names=self.reslist_dimer, res_ids=self.resid_list)

        return


    def move_double_att(self, a, b, extra_dels):
    
        # Define DNA box
        middnaA = (self.bond_pos[a] + self.bond_pos[a+1])/2
        dna_stericA = self.get_DNABox(self.dna_sbox, mid=middnaA)
        middnaB = (self.bond_pos[b] + self.bond_pos[b+1])/2 
        dna_stericB = self.get_DNABox(self.dna_sbox, mid=middnaB)
        
        new_xyzA, newcomA = align_two_molecules_at_two_points(self.xyzA, self.bond_pos[a], self.bond_pos[a+1], 
                                                              self.xyzA[self.attach_idxs[0]], self.xyzA[self.attach_idxs[1]], 
                                                              dna_stericA.com, self.comA)
        new_xyzB, newcomB = align_two_molecules_at_two_points(self.xyzB, self.bond_pos[b], self.bond_pos[b+1], 
                                                              self.xyzB[self.attach_idxs[0]], self.xyzB[self.attach_idxs[1]], 
                                                              dna_stericB.com, self.comB)
        
        # Appending current bond information (x position of an atom in the bonding residues)
        #  With double atachement, the res in-between is deleted
        pairs = self.bond_labels[a]+'&'+self.bond_labels[a+1]+'-'+self.bond_labels[b]+'&'+self.bond_labels[b+1]
        btwn =  [self.res_idx[a+1][0], self.res_idx[b-1][0]]
        if a+1 == b: # 0nt separation
            p1 = new_xyzA[self.attach_idxs[0]][0]
            o5 = new_xyzA[self.names=="O5'"][0][0]
            btwn = [np.round(p1,4), np.round(o5,4)]
        self.bonds.append([pairs, self.res_idx[a-1][0]] + btwn + [self.res_idx[b+1][0]])
        self.pairs.append(pairs)

        # Have to "manually" make the list of resnames and resids so it matches the mol2 files
        resname1 = self.reslist[0][0]
        resname2 = self.reslist[0][-1]
        self.reslist_dimer = [resname1, resname2]  
        natoms1, natoms2 = np.count_nonzero(
            self.reslist[0] == resname1), np.count_nonzero(self.reslist[0] == resname2)
        self.resid_list = [0]*natoms1 + [1]*natoms2
    
        self.dimer_mol = get_mol([new_xyzA, new_xyzB], [self.names]*2, [self.types]*2,
                                  res_names=self.reslist_dimer, res_ids=self.resid_list)
        
        self.del_res = f" and not (resname {self.bond_resn[a]} and resid {self.bond_resi[a]})"
        self.del_res +=  f" and not (resname {self.bond_resn[b]} and resid {self.bond_resi[b]})"

        for dels in extra_dels:
            self.dye_del += f"and not (resid 0 and name {dels})"
            self.dye_del += f" and not (resid 1 and name {dels})"
        return 

    def save_dimer_pdb(self, DNABox, path, idx, single_box):
        # Save dimer pdb with new coordinates

        dimer_mol = self.dimer_mol.select_atoms('all')
        pdb_name = f'{path}/dimer_{idx}.pdb'
        bond_name = f'{path}/dimer_{idx}.dat'

        if DNABox:
            nres = np.unique(self.reslist[0])
            if single_box: # A single box around dimer
                mid = dimer_mol.atoms.center_of_geometry()
            else:
                # Add a box around each molecule
                mid = [dimer_mol.select_atoms(f'resid 1 or resid {len(nres)}').atoms.center_of_geometry(),
                    dimer_mol.select_atoms(f'resid {len(nres)+1} or resid {len(nres)*2}').atoms.center_of_geometry()]
            nuc_box = self.get_DNABox(DNABox, mid)

            if nuc_box:
                combined = mda.Merge(
                    dimer_mol.atoms, nuc_box.atoms).select_atoms('all'+self.del_res+self.dye_del)
                # Delete incomplete nucleotides
                dimer_mol = del_extra_atoms(combined)
            else:
                print('!!! No DNA', idx, dimer_mol.atoms.center_of_geometry(), DNABox)

        dimer_mol.write(pdb_name)

        # 1) Find which residues are at the edges and save in file
        #edges = find_edge_residues(dimer_mol)
        #np.savetxt(bond_name, edges)
        return dimer_mol

    def check_valid_sample(self, a, b, dist_min):
        if self.num_attachs == 1:
            return LA.norm(self.bond_pos[a] - self.bond_pos[b]) < dist_min
        else:
            return LA.norm(self.bond_pos[a+1] - self.bond_pos[b]) < dist_min
        

    def find_edge_residues(atom_sel):
        """Helper function to return atoms on the edge of a DNA fragment
        Returns positions of P atoms (on left-edge residues) and O3' (right edge)

        Args:
            atom_sel (_type_): _description_

        Returns:
            _type_: _description_
        """
        reslist = atom_sel.select_atoms("nucleic").residues
        edge_atoms = []
        for res in reslist:
            if res.resid + 1 not in reslist.resids:
                redge = res.atoms.select_atoms("name O3*").positions
                if len(redge) > 0:
                    edge_atoms.append([res.resid]+list(redge[0]))
            if res.resid - 1 not in reslist.resids:
                ledge = res.atoms.select_atoms('name P').positions
                if len(ledge) > 0:
                    edge_atoms.append([res.resid]+list(ledge[0]))
        return edge_atoms


    def get_DNABox(self, DNABox_size, mid):

        """ Return a DNA box of size DNABox_size around the midpoint(s) coordinates
        Args:
            DNABox_size (float or int): Size of the DNA box (equal dist from center point)
            mid (np.ndarray or list): Center point(s). 
            Either an array with the coordinates of the single center, or a list of centers 
            (when we wish to return mutiple "boxes" around multiple centers).

        Returns:
            DNA class object with the nucleotide selection
        """

        
        if not isinstance(mid[0],np.ndarray):
            nuc_box = self.u_DNA.select_atoms(
                        f'point {mid[0]} {mid[1]} {mid[2]} {DNABox_size}')
            
        else:
            nuc_box = self.u_DNA.select_atoms(
                        f'point {mid[0][0]} {mid[0][1]} {mid[0][2]} {DNABox_size}' + 
                        f' or point {mid[1][0]} {mid[1][1]} {mid[1][2]} {DNABox_size}')
            
        nuc_box = self.DNA(nuc_box)
        
        if nuc_box.natoms > 0:
            return nuc_box
        else: 
            return None


def scan_DNAconfig(pdb_monomer, pdb_dna, path_save, resD, resL=None, 
                   chainDNA=None, dist_min=20, DNABox=20, DNASt=10,
                   attachement='single', attach_points=['P'], attach_prev=None,
                   add_deletes=None, single_box=False):
    """Scans all possible dimer configurations in the given DNA scaffold, for the given monomer pdb file.

    Args:
        pdb_monomer (string): Dir location of the monomer pdb to scan
        pdb_dna (string): Dir location of the DNA scaffold
        path_save (string): Folder to save samples
        resD (int): resid of the dye according to its pdb
        resL (int, optional): resid of the linker (if any) according to its pdb. Defaults to None.
        chainDNA (string, optional): segname of the DNA chain we wish to scan (if any). Defaults to None.
        dist_min (int, optional): The minimum distance the monomers should be (in A). Defaults to 20.
        DNABox (int, optional): The size of the DNA distance box to include with samples (in A). Defaults to 20.
        DNASt (int, optional): The size of the DNA steric box that should be accounted around a molecule (in A). Defaults to 10.
        attachement (str, optional): Whether the molecule has a 'single' attachement to DNA, or two ('double'). Defaults to 'single'.
        attach_points (list, optional): List with the atom names of the attachement points in the molecule. Defaults to ['P'].

    Raises:
        ValueError: Only 'single' or 'double' attachements are implemented.

    Returns:
        int: Number of samples generated
    """
 
    configs = scan_cofigs(type_of_attach=attachement, dna_stericBox=DNASt)

    configs.extract_monomer(pdb_monomer, resD, resL, attach_points, attach_prev) 
    configs.extract_DNA(pdb_dna, chain=chainDNA)

    print("atoms count", len(configs.dna.positions),
          len(configs.dna.chain_sel.atoms.positions))
    
    # Returns unique pairs for which distance is < dist_min
    all_pairs = 0
    idx = 0
    
    for i in range(1, len(configs.bond_pos)):
        for j in range(i+1, len(configs.bond_pos)-1):
            all_pairs += 1  # count total pairs just to check
            if configs.check_valid_sample(i, j, dist_min):
                # Move monA to point i, and monB to point B
                if attachement=='single':
                    configs.move_single_att(i, j)
                elif attachement=='double': 
                    if i+1 in configs.res_ter or j+1 in configs.res_ter:
                        continue
                    configs.move_double_att(i, j, extra_dels=add_deletes)
                else:
                    raise ValueError('The only valid attachements are "single" and "double"')

                # Save dimer pdb with new coordinates 
                configs.save_dimer_pdb(DNABox, path_save, idx, single_box)

                idx += 1

    # Print number of samples vs all possible
    print(f'All possible: {all_pairs}, valid pairs: {len(configs.pairs)}')

    # Save a file with all bonding info (which residues the monomers are attached to)
    # The X coordinate is saved for the DNA bonding atom
    with open(f'{path_save}/name_info.txt', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(configs.bonds)#zip(configs.pairs, configs.bonds))

    return len(configs.pairs)


def del_extra_atoms(u, st_atoms=25):
    """ Helper function to extra atoms left from using a spatial MDAnalysis selections
        (i.e., those belonging to an incomplete residue)
    Pending implementation!!
    """

    # 1) Select nucleotides and non DNA groups
    dna = u.select_atoms('nucleic')
    non_dna = u.select_atoms('not nucleic')
    # 2) If the res is a nucleotide (give this at input), check the total number of atoms
    total_valid = 0
    invalid_res = []
    for nuc in dna.residues:
        # if the atoms are complete, merge. "Complete" includes having both edge atoms.
        if len(nuc.atoms.select_atoms("name O3* or name P")) > 1 and len(nuc.atoms) >= st_atoms:
            non_dna = mda.Merge(non_dna.atoms, nuc.atoms)
            total_valid += 1
        else:
            invalid_res.append(nuc.resid)
    #print(f'A total of {len(dna.residues)-total_valid} incomplete residues were deleted')
    updated_unit = non_dna.select_atoms('all')

    # RETURN the updated atom selection with complete nucleotides (and use it in the scanDNA function.)
    return updated_unit


def scan_bond(molecule, fixed_a, fixed_b, fixed_length, nsamples, condition):
    """ Transform coordinates such that by rotating around a fixed_length bond between fixed_a(center) and fixed_b
        New coordinates are valid if they satisfy a given condition (function)

    Args:
        molecule (MDAnalysis AtomGroup): Molecule to transform
        fixed_a (numpy array, list): Coordinates of atom we wish to keep in place in bond (center of rot sphere)
        fixed_b (numpy array, list): Coordinates of atom we that is "rotating" in the bond
        fixed_length (Fixed bond length): The length of the bond we are scanning with resp with
        nsamples (int): How many point to sample on the sphere  
        condition (function(coords)): Functions defining condition by which transformed coords are valid (returns bool)

    Raises:
        ValueError: When none of the scanned tranformed coordinates were valid

    Returns:
        Numpy array: New transformed coordinates
    """

    # Define a sphere centered at fixed_a and whose radius is the length of the fixed bond
    # scan nsamples points in the surface of the sphere and define tvector wrt the fixed bond
    # then translate the entire molecule
    import random
    increment = 2*pi/nsamples
    offset = pi/nsamples

    # We randomize the samples to increase the chance of finding a valid transformation fast
    samples = random.sample(range(nsamples), nsamples)
    origin = fixed_a

    for i in range(nsamples):
        theta = i * increment
        for j in range(nsamples):
            phi = j * increment + offset
            sphere = np.array([sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)])
            sample_point = origin + fixed_length * sphere

            new_molecule = rotate_bond(molecule, fixed_a, fixed_b, sample_point)      

            #tvector = sample_point - fixed_b
            #new_molecule = molecule.atoms.translate(tvector)

            '''
            ref = fixed_a
            ref.atoms.positions = [origin,sample_point]
            new_molecule = align_to_mol(molecule, ref, path_save=None)
            '''

            # Test if configuration is valid
            if condition(new_molecule):
                return new_molecule

    #raise ValueError(
    print(f"A configuration wasn't found for the bond scan with {nsamples} samples. Increase nsamples!!")
    return new_molecule

def write_leap(mol2_dye, mol2_link, pdb, frcmod, file_save, dye_res='DYE', link_res='LNK',
               add_ions=None, make_bond=None, remove=None, water_box=20.0):
    """ Helper function to write tleap file

    Args:
        mol2_dye (_type_): _description_
        mol2_link (_type_): _description_
        pdb (_type_): _description_
        frcmod (_type_): _description_
        file_save (_type_): _description_
        dye_res (str, optional): _description_. Defaults to 'DYE'.
        link_res (str, optional): _description_. Defaults to 'LNK'.
        add_ions (_type_, optional): _description_. Defaults to None.
        make_bond (_type_, optional): _description_. Defaults to None.
        remove (_type_, optional): _description_. Defaults to None.
        water_box (float, optional): _description_. Defaults to 20.0.
    """
    fprefix = pdb[:-3]
    with open(file_save, 'w') as f:
        f.write("source leaprc.DNA.OL15\n")
        f.write("source leaprc.gaff\n")
        f.write("source leaprc.gaff2\n")
        f.write("source leaprc.water.tip3p\n")
        f.write(f"{dye_res} = loadmol2 {mol2_dye} \n")
        if mol2_link:
            f.write(f"{link_res} = loadmol2 {mol2_link} \n")
        f.write(f"loadAmberParams {frcmod} \n")
        f.write(f"mol = loadpdb {pdb} \n")
        if make_bond is not None:
            for m in make_bond:
                f.write(f"bond mol.{m[0]} mol.{m[1]} \n")
        if remove is not None:
            for r in remove:
                f.write(f"remove mol mol.{r} \n")
        if add_ions is not None:
            f.write(f"addIons mol {add_ions[0]} {add_ions[1]} \n")
        f.write(f"solvatebox mol TIP3PBOX {water_box}\n")
        f.write(f"saveAmberParm mol {fprefix}prmtop {fprefix}rst7\n")
        f.write("quit")

    return


def join_pdb(pdb_st, pdb_mob, del_st, del_mob, bond_st, bond_mob, path_bond_info=None, path_save=None):
    """Joining dye/linker pdb files with MDAnalysis

    Args:
        pdb_st (_type_): _description_
        pdb_mob (_type_): _description_
        del_st (_type_): _description_
        del_mob (_type_): _description_
        bond_st (_type_): _description_
        bond_mob (_type_): _description_
        path_bond_info (_type_, optional): _description_. Defaults to None.
        path_save (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # Get coordinates and label info   xyz, names, typ, CofM, resname
    xyzm, namesm, typm, CofMm, reslist = get_coords(pdb_mob, None, 0, sel_1=['1'], sel_2=None, cap=False)
    xyzs, namess, typs, CofMs, reslist = get_coords(pdb_st, None, 0, sel_1=['1'], sel_2=None, cap=False)
    xyzm, xyzs = xyzm[0], xyzs[0]
    namesm = namesm[0] 
    namess = namess[0] 
    typm, typs = typm[0], typs[0]
    CofMm, CofMs = CofMm[0], CofMs[0]

    # MDAnalysis atom indexes are zero-based
    del_st  = [i-1 for i in del_st]
    del_mob = [i-1 for i in del_mob]

    typm = del_atoms(typm, namesm, del_mob)
    res_names = ['DYE']
    if isinstance(bond_st[1], int):
        xyzm, CofMm = align_two_molecules_at_one_point(xyzm, xyzs[bond_st[1]-1], xyzm[bond_mob-1], 
                                                       CofMs, CofMm, align=1, accuracy=0.9)
        xyzm = del_atoms(xyzm, namesm, del_mob)
        xyzm = [xyzm]
        typm = [typm]
        res_names += ['LNK']
        # Saving info on atoms that participate in bonding (before deletion)
        bond_info = [namess[bond_st[0]-1], namesm[bond_mob-1]]
        namesm = del_atoms(namesm, namesm, del_mob)
        namesm = [namesm]

    else:
        xyzm1, CofMm1 = align_two_molecules_at_one_point(xyzm, xyzs[bond_st[0][1]-1], xyzm[bond_mob-1], 
                                                         CofMs, CofMm, align=1, accuracy=0.9)
        xyzm2, CofMm2 = align_two_molecules_at_one_point(xyzm, xyzs[bond_st[1][1]-1], xyzm[bond_mob-1],
                                                         CofMs, CofMm, align=1, accuracy=0.9)
        print('bonds',namesm[bond_mob-1], namess[bond_st[0][1]-1], namess[bond_st[1][1]-1])
        xyzm1 = del_atoms(xyzm1, namesm, del_mob)
        xyzm2 = del_atoms(xyzm2, namesm, del_mob)
        xyzm = [xyzm1, xyzm2]
        typm = [typm]*2
        res_names += ['LNK','LNK']
        # Saving info on atoms that participate in A and B bonding (before deletion)
        bond_info = [namess[bond_st[0][0]-1], namesm[bond_mob-1], namess[bond_st[1][0]-1], namesm[bond_mob-1]]
        namesm = del_atoms(namesm, namesm, del_mob)
        namesm = [namesm]*2    

    typs = del_atoms(typs, namess, del_st)
    xyzs = del_atoms(xyzs, namess, del_st)
    namess = del_atoms(namess, namess, del_st)
    typ = [typs] + typm 
    xyz = [xyzs] + xyzm
    names = [namess] + namesm
    

    if path_bond_info is not None:
        np.savetxt(path_bond_info, bond_info, fmt="%s")

    # Save final pdb
    dimer_mol = get_mol(xyz, names, typ, res_names=res_names)

    # save pdb
    dimer_all = dimer_mol.select_atoms('all')
    dimer_all.write(path_save)
    return bond_info


def find_idx(names, to_find, which):
    """Find indexes corresponding to given list of some characteristic
        e.g., names, coordinates, ...

    Args:
        names (_type_): _description_
        to_find (_type_): _description_
        which (_type_): _description_

    Returns:
        _type_: _description_
    """
    if type(to_find[0] is int):
        return to_find

    try:
        datom_idx = [np.nonzero(names == n)[0][0] for n in to_find]
    except:
        datom_idx = []
        print(
            f"The {to_find} couldn't be found. No atoms will be deleted in {which} molecule")
    return datom_idx


def del_atoms(var, ref, del_n):
    """Delete an atom from a variable var, given list of items from ref category

    Args:
        var (_type_): _description_
        names (_type_): _description_
        del_n (_type_): _description_

    Returns:
        _type_: _description_
    """
    del_idx = find_idx(ref, del_n, '')
    print('Deleting!',del_idx, ref[del_idx])
    var_new = np.delete(var, del_idx, 0)
    return var_new


def atom_dist(a1, a2, coord1, coord2):
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
    dist = LA.norm(coord2[a2] - coord1[a1])
    return dist


def check_steric(pos1, pos2, at1, at2, atH1, atH2):
    """_summary_

    Args:
        pos1 (numpy ndarray): Coordinates of mol 1
        pos2 (numpy ndarray): Coordinates of mol 2
        at1 (numpy ndarray): Atom indexes mol1
        at2 (numpy ndarray): Atom indexes mol2
        atH1 (numpy ndarray): Heavy atom indexes mol1
        atH2 (numpy ndarray): Heavy atom indexes mol2

    Returns:
        bool: If a steric clashes between non-H atoms are found
    """
    from scipy.spatial.distance import cdist

    if len(atH1) == 0 or len(atH2) == 0:
        return False

    atH1 -= at1[0]
    atH2 -= at2[0]

    at1 -= at1[0]
    at2 -= at2[0]

    # distsum = np.array([[atom_dist(a1,a2,pos1,pos2) for a2 in at2-1] for a1 in at1-1]) #array (natoms1 x natoms2)
    distsum = cdist(pos1, pos2)

    # Distance between non-H atoms
    # distsum_noH = np.array([[atom_dist(a1,a2,pos1,pos2) for a2 in atH2-1] for a1 in atH1-1]) #array (natoms1 x natoms2)
    pos_nH1 = pos1[atH1]
    pos_nH2 = pos2[atH2]
    distsum_noH = cdist(pos_nH1, pos_nH2)

    # Test if atoms are too close together (approx fails)
    # if np.count_nonzero((np.abs(distsum_noH) < 2.0)) > 5 or np.any((np.abs(distsum_noH) < 1.0)):
    if np.any((np.abs(distsum_noH) < 1.5)):  # molprobity clash score
        #print('tot = ', distsum[np.abs(distsum) < 2.0],', from:', len(at1),len(at2))
        return True # Means there's a clash
        # print(np.count_nonzero((np.abs(distsum_noH) < 2.0)))
    else:
        return False


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

    # distsum = np.array([[atom_dist(a1,a2,pos1,pos2) for a2 in at2-1] for a1 in at1-1]) #array (natoms1 x natoms2)
    distsum = cdist(pos1, pos2)

    # Distance between non-H atoms
    # distsum_noH = np.array([[atom_dist(a1,a2,pos1,pos2) for a2 in atH2-1] for a1 in atH1-1]) #array (natoms1 x natoms2)
    pos_nH1 = pos1[atH1]
    pos_nH2 = pos2[atH2]
    distsum_noH = cdist(pos_nH1, pos_nH2)

    # Test if atoms are too close together (approx fails)
    if np.count_nonzero((np.abs(distsum_noH) < 2.0)) > 5 or np.any((np.abs(distsum_noH) < 1.0)):
        #print('tot = ', distsum[np.abs(distsum) < 2.0],', from:', len(at1),len(at2))
        Vij = 9999999999999999
        # print(np.count_nonzero((np.abs(distsum_noH) < 2.0)))
    else:
        Vij = np.sum(np.outer(ch1, ch2)/distsum)
        # np.sum( np.multiply(np.outer(ch1,ch2),1/distsum) ) #SUM_{f,g}[ (qf qg)/|rf-rg| ]
        #print('!!',np.count_nonzero((np.abs(distsum_noH) < 2.0)), 'from:', len(at1),len(at2))

    return Vij


def get_mol(coords, names, types, res_names, segname='1', res_ids=None):
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
    segname : string
    res_ids : list
    Returns
    -------
    mol_new : MDAnalisys.AtomGroup
        Transformed molecule object.
    """
    if not len(coords) == len(names) == len(types):  # == len(res_names):
        raise ValueError("All input arrays must be of length Nmolecules")

    n_residues = len(res_names)
    n_mols = len(names)
    # Creating new molecules
    resids0 = []
    natoms = 0
    if res_ids is None:
        for imol in range(n_mols):
            natom = len(names[imol])
            resid = [imol]*natom
            resids0.append(resid)
            #natoms += natom
        resids = np.concatenate(tuple(resids0))
    else:
        resids = res_ids
    natoms = 0
    for imol in range(len(names)):
        natoms += len(names[imol]) 

    assert len(resids) == natoms
    segindices = [0] * n_residues

    atnames = np.concatenate(names, axis=0)#tuple(names))
    attypes = np.concatenate(types, axis=0)#tuple(types))
    # if isinstance(res_names[0], str):
    resnames = res_names
    # else:
    #resnames = np.concatenate(tuple(res_names))

    mol_new = mda.Universe.empty(natoms,
                                 n_residues=n_residues,
                                 atom_resindex=resids,
                                 residue_segindex=segindices,
                                 trajectory=True)

    mol_new.add_TopologyAttr('name', atnames)
    mol_new.add_TopologyAttr('type', attypes)
    mol_new.add_TopologyAttr('resname', resnames)
    mol_new.add_TopologyAttr('resid', list(range(1, n_residues+1)))
    mol_new.add_TopologyAttr('segid', [segname])
    mol_new.add_TopologyAttr('id', list(range(natoms)))
    mol_new.add_TopologyAttr('record_types', ['HETATM']*natoms)

    # Adding positions
    coord_array = np.concatenate(coords, axis=0) #np.concatenate(tuple(coords))
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
    neg_coords = []
    neg_charges = np.array([neg_charge] * neg_res.residues.n_residues)
    for res in neg_res.residues:
        cofm = res.atoms.center_of_geometry()
        neg_coords.append(cofm)

    coords = np.concatenate((dna_coords, pos_coords, neg_coords), axis=0)
    charges = np.concatenate((dna_charges, pos_charges, neg_charges), axis=0)
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
        def param_files(ty): return 'Sq' + ty + '_dimer_g1.prmtop'
    elif rep == 'D':
        def param_files(ty): return 'Sq' + ty + '_dimer_g2.prmtop'
    else:
        def param_files(ty): return 'Sq' + ty + '_dimer.prmtop'

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
            t_i = round((ts.frame*10), 2)
            print("The time-step is: ", t_i)

            coord, charge = get_charges(u, ("MG", 2.0), ("Cl-", -1.0))

    return coord, charge


def get_pdb(traj_path, param_path, path_save, resnums, select=(0, 0), dt=2, MDA_selection='all',
            del_list=[], cap_list=[[], []], resnames=['MOA', 'MOB'], mol_name='ABC'):

    i_file, idx = select

    xyza, xyzb, namesA, namesB, typeA, typeB, __, __ = get_coords(traj_path, param_path,
                                                                  idx, file_idx=i_file, dt=dt,
                                                                  sel_1=resnums[0], sel_2=resnums[1],
                                                                  del_list=del_list, cap_list=cap_list)

    orig_mol = get_mol([xyza, xyzb], [namesA, namesB], [
                       typeA, typeB], res_names=resnames, segname=mol_name)

    # save pdb
    orig_all = orig_mol.select_atoms(MDA_selection)
    orig_all.write(path_save)
    return orig_all


def max_pdb(Vi, traj_path, param_path, resnums, path_save, sel='all'):
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

    obj = get_pdb(traj_path, param_path, path_save, resnums, select=(i_file, idx),
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
    energies = np.empty((0, 2))
    for f in file_list:
        data = np.loadtxt(f)
        energies = np.concatenate((energies, data))

    energies = eV_conv*(energies-global_min)

    return energies


def displaced_dimer(sel1, sel2, cof_dist, disp,
                    atom_orig='N1', atom_z='N2', atom_y='C2',
                    res_names=['SQA', 'SQB']):

    xyz1 = sel1.positions
    xyz2 = sel2.positions

    idx1A = np.nonzero(sel1.atoms.names == atom_orig)[0][0]
    idx1B = np.nonzero(sel1.atoms.names == atom_z)[0][0]
    idx1C = np.nonzero(sel1.atoms.names == atom_y)[0][0]
    idx2A = np.nonzero(sel2.atoms.names == atom_orig)[0][0]
    idx2B = np.nonzero(sel2.atoms.names == atom_z)[0][0]
    idx2C = np.nonzero(sel2.atoms.names == atom_y)[0][0]

    x_unit = np.array([1, 0, 0]).reshape(1, -1)
    y_unit = np.array([0, 1, 0]).reshape(1, -1)
    z_unit = np.array([0, 0, 1]).reshape(1, -1)

    # Rodrigues formula
    def rodrigues(xyz, phi, v_unit):

        xyz_new = ((xyz.T*np.cos(phi)).T + np.cross(v_unit[0, :], xyz.T, axis=0).T * np.sin(phi)
                   + (v_unit * np.dot(v_unit, xyz.T).reshape(-1, 1)) * (1-np.cos(phi)))

        return xyz_new

    # placing atomA of both molecules in the origin
    xyz1 -= xyz1[idx1A]
    xyz2 -= xyz2[idx2A]

    # rotating molecule to yz plane
    rot1 = xyz1[idx1B]
    #phiz1 = - np.arccos(rot1[2]/LA.norm(rot1))
    num = rot1[1]*np.sqrt(rot1[0]**2+rot1[1]**2) + rot1[2]**2
    theta1 = -np.arccos(num/LA.norm(rot1)**2)
    xyz1_new = rodrigues(xyz1, theta1, z_unit)

    # rotating to z axis
    rot1 = xyz1_new[idx1B]
    phi1 = np.arccos(rot1[2]/LA.norm(rot1))
    xyz1_new = rodrigues(xyz1_new, phi1, y_unit)

    # Rotating the other atom axis
    rot1 = xyz1_new[idx1C]
    psi1 = np.arccos(rot1[1]/LA.norm(rot1))
    xyz1_new = rodrigues(xyz1_new, psi1, x_unit)

    ###
    rot2 = xyz2[idx2B]
    num = rot2[1]*np.sqrt(rot2[0]**2+rot2[1]**2) + rot2[2]**2
    theta2 = -np.arccos(num/LA.norm(rot2)**2)
    xyz2_new = rodrigues(xyz2, theta2, z_unit)

    # rotating to z axis
    rot2 = xyz2_new[idx2B]
    phi2 = np.arccos(rot2[2]/LA.norm(rot2))
    xyz2_new = rodrigues(xyz2_new, phi2, y_unit)

    # Rotating the other atom axis
    rot2 = xyz2_new[idx2C]
    psi2 = np.arccos(rot2[1]/LA.norm(rot2))
    xyz2_new = rodrigues(xyz2_new, psi2, x_unit)

    #xyz2_new -= xyz2_new[idx2A]

    # displacing sel2 on the x axis only
    xyz2_new[:, 0] += cof_dist

    # displacing sel2 on y axis
    xyz2_new[:, 2] += disp

    # create new MDA object
    mol = get_mol([xyz1_new, xyz2_new], [sel1.atoms.names, sel2.atoms.names],
                  [sel1.atoms.types, sel2.atoms.types], [
                      sel1.atoms.bonds, sel2.atoms.bonds],
                  res_names=res_names)

    # print(rot1[2]*LA.norm(rot1))

    return mol





def calc_displacement(path, param_file, atom1, atom2, file_idx=None, dt=2, sel_1=['11'], sel_2=['12']):

    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str

    if path[-2:] == 't7':
        # To get universe from a rst7 file
        u = mda.Universe(param_file, path, format="RESTRT")
        dt = 1
    elif path[-2:] == 'db':
        # To get universe from a pdb file
        u = mda.Universe(path, format="PDB")
        dt = 1
    else:
        # To get universe form a nc trajectory
        traj_file = path
        fidx = str(file_idx) if file_idx is not None else ''
        print("The param file is: %s \n" % (param_file),
              "And the traj files is: ", traj_file+fidx+".nc")
        u = mda.Universe(param_file, traj_file+fidx+".nc")

    for fi, ts in enumerate(u.trajectory[::dt]):

        agA = u.select_atoms(make_sel(sel_1) + " and name " + atom1)
        agB = u.select_atoms(make_sel(sel_2) + " and name " + atom1)

        # Calculating displacement
        xyzA = agA.positions
        xyzB = agB.positions
        disp = abs(LA.norm(xyzA - xyzB))

    return disp


def pa_angle(molA, molB, pa=2):
    from numpy import pi
    mol1 = recenter_mol(molA, align_vec=None)
    mol2 = recenter_mol(molB, align_vec=None)
    paA = mol1.principal_axes()[pa]
    paB = mol2.principal_axes()[pa]
    thetaAB = np.arccos(np.dot(paA, paB)/(LA.norm(paA)*LA.norm(paB)))
    if thetaAB/pi > 0.5:
        thetaAB = pi - thetaAB

    return thetaAB


def calc_angle(path, param_file, file_idx=None, dt=2, sel_1=['11'], sel_2=['12'], pa=2):

    def make_sel(sel):
        sel_str = "resid "
        for st in range(len(sel)):
            sel_str += str(sel[st])
            if sel[st] != sel[-1]:
                sel_str += " or resid "
        return sel_str

    if path[-2:] == 't7':
        # To get universe from a rst7 file
        u = mda.Universe(param_file, path, format="RESTRT")
        dt = 1
    elif path[-2:] == 'db':
        # To get universe from a pdb file
        u = mda.Universe(path, format="PDB")
        dt = 1
    else:
        # To get universe form a nc trajectory
        traj_file = path
        fidx = str(file_idx) if file_idx is not None else ''
        print("The param file is: %s \n" % (param_file),
              "And the traj files is: ", traj_file+fidx+".nc")
        u = mda.Universe(param_file, traj_file+fidx+".nc")

    angles = []
    for fi, ts in enumerate(u.trajectory[::dt]):

        agA = u.select_atoms(make_sel(sel_1))
        agB = u.select_atoms(make_sel(sel_2))

        # Calculating angle
        thetaAB = pa_angle(agA, agB, pa)
        angles.append(thetaAB)

    return np.array(angles)


def recenter_mol(mol, align_vec=[1, 0, 0]):
    '''
    Re centers molecule to the origin and aligns its long axis.
    Parameters
    ----------
    mol : MDAnalysis object
    align_vec : array, optional
        vector to align long axis of mol to. The default is [1,0,0]: the x axis.

    Returns
    -------
    mol : TYPE
        DESCRIPTION.
    '''
    xyz = mol.atoms.positions
    cofm = mol.center_of_mass()
    new_xyz = xyz - cofm
    mol.atoms.positions = new_xyz

    if align_vec is not None:
        paxis = 2  # Aligning the long axis of the molecule
        mol = mol.align_principal_axis(paxis, align_vec)
    return mol


def align_to_mol(mol, ref, path_save=None):
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    if type(mol) is str:
        mol = mda.Universe(mol, format="PDB")
    if type(ref) is str:
        ref = mda.Universe(ref, format="PDB")
    align.alignto(mol, ref, select="all", weights="mass")

    mol = mol.select_atoms('all')
    if path_save:
        mol.write(path_save)
    return mol

def rotation_to_vector(mobile_vector, ref_vector):
    """Returns the rotation matrix required to align mobile_vector with ref_vector

    Args:
        mobile_vector (_type_): vector to be rotated
        ref_vector (_type_): reference vector
    """

    from scipy.spatial.transform import Rotation as R

    # Normalize 
    mobile_vector = mobile_vector / LA.norm(mobile_vector)
    ref_vector = ref_vector / LA.norm(ref_vector)

    # Axis of rotation
    k = np.cross(ref_vector, mobile_vector)
    k /= LA.norm(k)

    # Angle of rotation
    dot = np.dot(ref_vector, mobile_vector)
    angle = np.arccos(dot) # Given the vectors are normalized

    rotvec = -k*angle
    rotation_mat = R.from_rotvec(rotvec)

    return rotation_mat

def align_two_molecules_at_one_point(mol, pt1, pt2, com1, com2, align=-1, accuracy=0.5):

    from scipy.spatial.transform import Rotation as R

    # rotate mol to align line segments: Find R such that pta2 vector aligns with pta1 vector    
    ## Axis of rotation
    def rotate_around_segment(mol, pt, com_mol, com_fixed):
        axis1 = np.cross(com_mol, com_fixed)
        axis1 /= LA.norm(axis1)
        dot = np.dot(com_mol, com_fixed)
        angle2 = pi - np.arccos(dot/(LA.norm(com_fixed)*LA.norm(com_mol))) 
        rotvec2 = axis1*angle2 
        R2 = R.from_rotvec(rotvec2)
        mol_shifted = R2.apply(mol)
        com_shifted = R2.apply(com_mol)
        pt_shifted = R2.apply(pt)
        return mol_shifted, com_shifted, pt_shifted

    # The com1 and com2 vectors point in opp directions if the dot product 
    #  between the normalized vetors is approx 1, so we have to repeat the rotation until that's true.

    attempts = 0
    mol_shifted, com2_shifted, pt2_shifted = rotate_around_segment(mol, pt2, com2, com1)
    while align*np.dot(com2_shifted/LA.norm(com2_shifted), com1/LA.norm(com1))<accuracy and attempts<10:
        mol_shifted, com2_shifted, pt2_shifted = rotate_around_segment(mol_shifted, pt2_shifted, com2_shifted, com1)
        attempts += 1

    # translate mol to mol1

    disp = -pt2_shifted + pt1
    mol_shifted += disp
    com2_shifted += disp
    pt2_shifted += disp

    return mol_shifted, com2_shifted

def align_two_molecules_at_two_points(mol, pt1a, pt1b, pt2a, pt2b, com1, com2, align=-1, accuracy=0.5):
    """ Align mol so that the line segment (pt1a, pt1b) in the frame of reference for the mol
    overlaps with the line segment (pt2a, pt2b) in the frame of reference for an static molecule.
    The RMS distance between the alignment points is minimized,
    & the distance between the COM of each molecule is maximized.

    Args:
        mol  (numpy array): coordinates of mol we wich to align
        pt1a (numpy array): coordinates of 1st attachement target point
        pt1b (numpy array): coordinates of 2nd attachement target point
        pt2a (numpy array): coordinates of 1st attachement mobile point
        pt2b (numpy array): coordinates of 2nd attachement mobile point
        com1 (numpy array): center of masss of static object
        com2 (numpy array): center of masss of mobile object

    Returns:
        numpy array: aligned coordinates of mol2
    """


    from scipy.spatial.transform import Rotation as R

    # translate line segments to the origin
    shift1 = 0.5*(pt1a + pt1b) #midpoint
    shift2 = 0.5*(pt2a + pt2b)
    pt1a_shifted = pt1a - shift1
    pt2a_shifted = pt2a - shift2
    com1_shifted = com1 - shift1 
    com2_shifted = com2 - shift2
    mol_shifted = mol - shift2

    # rotate mol to align line segments: Find R such that pta2 vector aligns with pta1 vector    
    ## Axis of rotation
    axis1 = np.cross(pt1a_shifted, pt2a_shifted)
    axis1 /= LA.norm(axis1)

    # Calculate angle between the two segments: Formula α = arccos[(a · b) / (|a| * |b|)] 
    dot = np.dot(pt1a_shifted, pt2a_shifted)
    angle1 = np.arccos(dot/(LA.norm(pt1a_shifted)*LA.norm(pt2a_shifted)))
    rotvec1 = -axis1*angle1
    R1 = R.from_rotvec(rotvec1)
    # Rotate molecule 2
    mol_shifted = R1.apply(mol_shifted)
    pt2a_shifted = R1.apply(pt2a_shifted)
    com2_shifted = R1.apply(com2_shifted)

    # rotate mol around the pta2 axis, such that COM2 is aligned opposite to COM1 (minimizes steric)
    def rotate_around_segment(mol, com_mol, com_fixed, axis):
        dot = np.dot(com_mol, com_fixed)
        angle2 = pi - np.arccos(dot/LA.norm(com_fixed)/LA.norm(com_mol)) # antiparallel algment
        rotvec2 = axis*angle2*align
        R2 = R.from_rotvec(rotvec2)
        mol_shifted = R2.apply(mol)
        com_shifted = R2.apply(com_mol)
        return mol_shifted, com_shifted

    # The com1,com2 vectors point in opp directions if the dot product between the normalized vetors is approx -1.
    #  We can calculate the rotation angle but the final rotated vector depends on the sign of the rotation. 
    #  Idk how to verify the rotation direction is correct in each case, so we repeat the rotation until they're approx opp.
    attempts = 0
    axis2 = pt2a_shifted/LA.norm(pt2a_shifted)
    mol_shifted, com2_shifted = rotate_around_segment(mol_shifted,com2_shifted, com1_shifted, axis2)
    while align*np.dot(com2_shifted/LA.norm(com2_shifted), com1_shifted/LA.norm(com1_shifted))<accuracy and attempts<10:
        attempts += 1
        mol_shifted, com2_shifted = rotate_around_segment(mol_shifted, com2_shifted, com1_shifted, axis2)

    # translate mol to mol1
    mol_shifted += shift1

    return mol_shifted, com2_shifted

def rotate_bond(molecule, fixed_atom, mobile_atom, target_atom):
    """ Rotates a molecule wrt to a bond, such that mobile_atom is aligned with target_atom

    Args:
        molecule (_type_): _description_
        fixed_atom (_type_): Coordinates of atom in the bond to remain fixed in rotation
        mobile_atom (_type_): Coordinates of atom we want to move to target
        target_atom (_type_): Coordinates of target atom
    """
    # Calculate the vectors
    mobile_vector = mobile_atom - fixed_atom
    reference_vector = target_atom - fixed_atom

    rot_mat = rotation_to_vector(mobile_vector, reference_vector)

    molecule.atoms.rotate(rot_mat)

    return molecule


def displace_dimer(mol1, mol2, inter_dist, h_disp, v_disp, path_save,
                   inter_ax=1, long_axis=0, short_ax=2, resnames=['A', 'B'], mol_name='ABC'):

    align_vec = [0]*3
    align_vec[long_axis] = 1
    # Re-center to origin and align paxis to given axis
    mol1_0 = recenter_mol(mol1, align_vec=align_vec)
    mol2_0 = recenter_mol(mol2, align_vec=align_vec)

    xyz1 = mol1_0.atoms.positions
    xyz2 = mol2_0.atoms.positions
    names1 = mol1_0.atoms.names
    names2 = mol2_0.atoms.names
    types1 = mol1_0.atoms.types
    types2 = mol2_0.atoms.types

    # Distance between molecules
    xyz2[:, inter_ax] += inter_dist
    # Vertical displacement
    xyz2[:, short_ax] += v_disp
    # Horizontal displacement
    xyz2[:, long_axis] += h_disp

    new_mol = get_mol([xyz1, xyz2], [names1, names2], [types1, types2],
                      res_names=resnames, segname=mol_name)
    dimer = new_mol.select_atoms('all')
    dimer.write(path_save)

    return dimer


def rotation_mat(angle, rot_axis):
    from numpy import sin, cos
    '''
    Rx = np.array([[1, 0, 0],
                   [0, cos(angle), sin(angle)],
                   [0,-sin(angle), cos(angle)]])
    Ry = np.array([[cos(angle),0,-sin(angle)],
                   [0, 1, 0],
                   [sin(angle),0, cos(angle)]])
    Rz = np.array([[cos(angle),sin(angle) ,0],
                   [-sin(angle), cos(angle), 0],
                   [0 ,0, 1]])
    '''
    if rot_axis == 'x':
        return Rx(angle)
    elif rot_axis == 'y':
        return Ry(angle)
    else:
        return Rz(angle)


def vec_two_atoms(u, sel, atom1, atom2):
    ag1 = u.select_atoms(f"resid {sel} and name {atom1}")
    ag2 = u.select_atoms(f"resid {sel} and name {atom2}")
    p1 = np.array(ag1.positions[0])
    p2 = np.array(ag2.positions[0])
    return p1-p2


def angle_two_vectors(u1, u2, selA, selB, atoms_vec=['N', 'N1']):
    '''
       Formula α = arccos[(a · b) / (|a| * |b|)] 
    '''

    vecA = vec_two_atoms(u1, selA, atoms_vec[0], atoms_vec[1])
    vecB = vec_two_atoms(u2, selB, atoms_vec[0], atoms_vec[1])

    dot = np.dot(vecA, vecB)
    mult = LA.norm(vecA) * LA.norm(vecB)
    angle = np.arccos(dot/mult)

    return angle


def com_distance(u1, u2, selA, selB):
    '''
       Center of mass distance between to universe objects
    '''

    ag1 = u1.select_atoms(f"resid {selA}")
    ag2 = u2.select_atoms(f"resid {selB}")

    CofMA = ag1.center_of_mass()
    CofMB = ag2.center_of_mass()

    rab = abs(CofMA-CofMB)

    return LA.norm(rab)
