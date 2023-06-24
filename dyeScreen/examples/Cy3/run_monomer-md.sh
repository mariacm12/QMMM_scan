#!/bin/sh
#SBATCH --job-name="dyelink"
#SBATCH --nodes=1
#SBATCH --output=dyelink_FF.log

# Define variables
# (Replace paths when needed)
AMBER="~/miniconda3/envs/AmberTools23"
SANDER="~/miniconda3/envs/AmberTools23/bin/sander"
PARAM="cy3_dyelink/dye-with-linker.prmtop" # Not an actual file, it's during FF creation process 
INPUT1="../../MD/FF_min1.in"
INPUT2="../../MD/FF_min2.in"
SUMM1="cy3_dyelink/Pent_min1.out"
SUMM2="cy3_dyelink/Pent_min2.out"
REF="cy3_dyelink/dye-with-linker.rst7" 

# Generate min1 file 

COORDI="cy3_dyelink/dye-with-linker.rst7"
COORDF="cy3_dyelink/dye-with-linker.ncrst"

# Example on how to define content of first bash script
SCRIPT_CONTENT="#!/bin/sh\n#SBATCH --job-name=min1\n#SBATCH --partition=gpu\n#SBATCH --nodes=1\n#SBATCH --output=out.log\n\
\n$SANDER -O -i $INPUT1 -o $SUMM1 -p $PARAM -c $COORDI -r $COORDF -ref $REF\n\
wait\n"

# Write min1 bash script to file
echo "$SCRIPT_CONTENT" > run_min1.sh
chmod +x run_min1.sh

COORDI="cy3_dyelink/dye_min1.ncrst"
COORDF="cy3_dyelink/dye_min2.ncrst"

# Example on how to Define content of second bash script
SCRIPT_CONTENT="#!/bin/sh\n#SBATCH --job-name=min2\n#SBATCH --output=out2.log\n\
\n$SANDER -O -i $INPUT2 -o $SUMM2 -p $PARAM -c $COORDI -r $COORDF -ref $REF\n\
wait\n"

# Write min1 bash script to file
echo "$SCRIPT_CONTENT" > run_min2.sh
chmod +x run_min2.sh

JOBID1=$(sbatch --parsable run_min1.sh)
# Submit second job as a dependent job
sbatch --dependency=afterok:$JOBID1 run_min2.sh

rm run_min1.sh 
rm run_min2.sh

# Extract monomer from the last frame after min2, stripping water and ions
# Generate cpptraj input

FINAL_PDB="cy3_dyelink/dye_monomer.pdb"
SCRIPT_CONTENT="parm $PARAM\ntrajin $COORDF lastframe\nautoimage\n\
strip :WAT,NA outprefix stripped\n\
trajout ${FINAL_PDB}.pdb"
echo "$SCRIPT_CONTENT" > get_min-monomer.in
#run cpptraj
source $AMBER/amber.sh
$AMBER/bin/cpptraj -i get_min-monomer.in

# cpptraj generates atoms as ATOM instead of HETATM so have to replace
old_string="ATOM  "
new_string="HETATM"

# Use sed to replace the old string with the new string
sed -i "s/${old_string}/${new_string}/g" ${FINAL_PDB} 

# Delete the extra atoms in the phospate to be replaced by DNA bond
s="O4  LNK"
sed -i "/${s}/d" ${FINAL_PDB}
s="O5  LNK"
sed -i "/${s}/d" ${FINAL_PDB}
s="O6  LNK"
sed -i "/${s}/d" ${FINAL_PDB}
s="H20 LNK"
sed -i "/${s}/d" ${FINAL_PDB}
