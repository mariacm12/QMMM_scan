import numpy as np
import dyeScreen.commons.geom_utils as gu
import MDAnalysis as mda
import csv
import subprocess


# -6) Scan DNA configs 

amber_path = "/UserHome/opt/miniconda3/envs/AmberTools23/bin/" # change if different
path = "examples/Cy3/cy3_dyelink"

leap_out = path + "tleap_screen.in"
frcmod = path + "cy3_final.frcmod"
dye_mol2 = path + "cy3-link.mol2"
dye_pdb = path + "cy3-link.pdb"

DNA_pdb =  "examples/dna_duplex.pdb"

attach_pattern = ['P', 'O2']
# Sample only the positions within 12A of separation and include a box of DNA of 20A
nsamples = gu.scan_DNAconfig(dye_pdb, DNA_pdb, path+"samples", resD=1, resL=None, 
                             chainDNA=None, dist_min=12, DNABox=35, DNASt=20,
                             attachement='double', attach_points=['P1','P2'], 
                             attach_prev=['O1','O2'], add_deletes=['O3','O5']) 

attach_cy3 = ['P1', "O5'"]*2
attach_dna = ["O3'", "P"]*2

# Loop through the samples
bonds = open(path+"samples/name_info.txt")
lines = csv.reader(bonds, delimiter='\t')
for i, line in enumerate(lines):

    # 1) Clean the pdb samples
    subprocess.run(f"sed -i '/P2  CY3/d' {path}samples/dimer_{i}.pdb".split(' '))
    subprocess.run(f"{amber_path}pdb4amber -i {path}samples/dimer_{i}.pdb -o {path}samples/dimer_{i}_clean.pdb".split(' '))

    # 2) Search for the new res ids of the DNA nucleotides bonding to the monomers
    pdb_name = f'{path}samples/dimer_{i}_clean.pdb'
    u = mda.Universe(pdb_name, format="PDB")
    posX = u.atoms.positions[:,0]
    resids = u.atoms.resids

    make_bond = []
    res = 1
    for b, bond in enumerate(line[1:]):
        res = b//2 + 1
        res_loc = resids[np.argwhere(abs(posX - float(bond))<0.001)[0]][0]
        make_bond.append([f"{res}.{attach_cy3[b]}",f"{res_loc}.{attach_dna[b]}"])

    # 3) Create leap files 
    gu.write_leap(dye_mol2, None, f'{path}samples/dimer_{i}_clean.pdb', frcmod, f'{path}samples/dimer_{i}.in', 
                dye_res='PNT', add_ions=['NA', '0'],
                make_bond=make_bond, water_box=20.0)

    # Use sed to delete lines starting with "CONECT"
    subprocess.run(f"sed -i '/^CONECT/d' {path}samples/dimer_{i}_clean.pdb".split(' '))

subprocess.run(f"rm {path}samples/*sslink*".split(' '))
subprocess.run(f"rm {path}samples/*renum*".split(' '))
subprocess.run(f"rm {path}samples/*nonprot*".split(' '))

print('** Final',i)


## Pending: Currently double attachemnt is specific to the no-linker case. Generalize functions to support dye+linker 


