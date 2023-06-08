# QMMM_scan
Code for studying the Electronic Structure properties of chromophores in dissordered systems. 
Provides functions for studying the molecules based on Molecular Dynamics (MD) simulations.

- **couplingutils**: Functions for carrying out Quantum Mechanical (QM) calculations from MD trajectory coordinates. 
All QM is performed in PySCF, MD trajectories are process with MDAnalaysis.
- **geom_utils**: FUnctions for geometry manipulation/processing from MD trajectories or PDB files data. Integration is done using MDAnalyisis. 
- **examples/example_properties**: Example of using couplingutils to calculate the electronic coupling, transition energy and oscillator strength of chromophore dimers given traj or pdb files.
