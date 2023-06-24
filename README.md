# QMMM_scan
Code for studying the Electronic Structure properties of chromophores in dissordered systems. 
&  High-Throughput Screening of Molecules in DNA Scaffolding (*in the works*)
Provides functions for studying the molecules based on Molecular Dynamics (MD) simulations.

**Inside the dyeScreen folder:**

-**`commons/couplingutils`**: Functions for carrying out Quantum Mechanical (QM) calculations from MD trajectory coordinates. 
All QM is performed in PySCF, MD trajectories are process with MDAnalaysis.
- **`commons/geom_utils`**: Functions for geometry manipulation/processing from MD trajectories or PDB files data. Integration is done using MDAnalyisis. 
- **`examples/example_properties`**: Example of using couplingutils to calculate the electronic coupling, transition energy and oscillator strength of chromophore dimers given traj or pdb files.
- **`examples/Cy3`** (*in progress*): Example on how to construct a FF for a dye (Cy3) with a linker to be scaffolded in DNA, screening all valid configurations within a DNA structure, and running
MD simulations on Amber for all samples. ***Pending***: QM to analyze MD results of all samples. 

**\*\*Pre-requisites: Working PySCF and MDAnalysis installation**
