## Files
- **`Cy3/run_cy3-dyelink.py`**: Starting from a geometry optimized Cy3 and linker pdb, prepares a FF for the Cy3 + 2linkers
     returns the input file for the tleap run (Amber) and also the FF files: mol2(linker & dye), pdb(linker+dye), frcmod(linker+dye)   
- **`Cy3/gen_configs_Cy3.py`**: Scan the configurations for a Cy3 dimer in a DNA duplex. PDB and tleap input files are saved in the folder `samples/`
- **`example_properties`**: Example of using couplingutils to calculate the electronic coupling, transition energy and oscillator strength of chromophore dimers given traj or pdb files.

## Examples
Results are saved in folders:
- *`cy3_dyelink/`*
Each will contain all output files and samples from each corresponding run script. 

## Pendings
***frcmod in the dye + linker FF construction***
Joining of frcmod files is manual. Code returns separate frcmod's for dye and linker, but have to manually copy/paste.
***Automatic MD runs for dye + linker samples in construction***

