import numpy as np
import subprocess
import MDAnalysis as mda

class md_samples():
    def __init__(self, amber_path, nres, file_prefix, cutoff=12.0, sander_path=None, edge_res=None, edge_rest=10.0) -> None:
        self.amber = amber_path
        self.sander = sander_path
        if self.sander == None:
            self.sander = self.amber + "/bin/sander"
        self.f_prefix = file_prefix
        self.cutoff = cutoff
        self.nres = nres
        # Optionally constrain DNA edges
        res_str = ""
        if len(edge_res)>0:
            res_str = f'\n  restraint_wt = {edge_rest},     ! kcal/mol-A**2 force constant\n'
            res_list = ",".join(map(str, edge_res))
            res_str += f"  restraintmask = '(:{res_list})',"
        self.edgeRest = res_str
        # Standard harmonic restrains
        self.min_rest = 500.0
        self.eq_rest = 10.0
        # Standard temperature
        self.rtemp = 300.0

    def make_min_input(self, rest, edge_str, ncycles):
        ''' Sets up the Amber files for the constrained & unconstrained minimization
        '''

        # Restrains
        ntr_i = 0
        if edge_str:
            ntr_i = 1

        min1_ncycles, min2_ncycles = ncycles[0], ncycles[1]
        file_min1 = self.f_prefix + "_min1.in"
        file_min2 = self.f_prefix + "_min2.in"
        clean_prefix = self.f_prefix.split("/")[-1]
        print(clean_prefix)

        with open(file_min1, 'w') as f:
            f.write(f"{clean_prefix}: Initial minimization of solvent + ions\n")
            f.write(" &cntrl\n")
            f.write(f'''  imin   = 1,
  maxcyc = {min1_ncycles+2000},
  ncyc   = {min1_ncycles},
  ntb    = 1,
  ntr    = 1,
  iwrap  = 1,
  cut    = {self.cutoff}
/
Hold the DNA fixed
{rest}
RES 1 {self.nres}''')
            f.write('\nEND\nEND')

        with open(file_min2, 'w') as f:
            f.write(f"{clean_prefix}: Initial minimization of entire system\n")
            f.write(" &cntrl\n")
            f.write(f'''  imin   = 1,
  maxcyc = {min2_ncycles+2000},
  ncyc   = {min2_ncycles},
  ntb    = 1,
  ntr    = {ntr_i},
  iwrap  = 1,{edge_str}
  cut    = {self.cutoff},\n/\n''')

        return file_min1.split("/")[-1], file_min2.split("/")[-1]

    def make_heat_input(self, tF, rest, total_ps, dt, twrite):
        ''' Sets up the Amber file for the constrained heating
        '''
        nsteps = int(total_ps/dt) 
        nsteps_x = int(twrite/dt) # How often to print trajectory
        file_heat = self.f_prefix + "_eq1.in"
        clean_prefix = self.f_prefix.split("/")[-1]

        rest_str = ""
        with open(file_heat, 'w') as f:
            f.write(f"{clean_prefix}: {total_ps}ps MD equilibration with restraint on DNA\n")
            f.write(" &cntrl\n")
            f.write(f'''  imin   = 0,
  irest  = 0,
  ntx    = 1,
  ntb    = 1,
  iwrap  = 1,
  cut    = {self.cutoff}, !non-bonded cutoff in Ams
  ntr    = 1,
  ntc    = 2,
  ntf    = 2,
  tempi  = 0.0,
  temp0  = {tF},
  ntt    = 3,
  gamma_ln = 5.0,
  ig=-1,
  nstlim = {int(nsteps)},
  dt = {dt},
  ntpr = {nsteps_x}, !How often to write in out file
  ntwx = {nsteps_x}, !How often to write in traj file
  ntwr = {nsteps_x}, !How often to write in restart file
  ioutfm = 0
/
Hold the DNA fixed with weak restrains
{rest}
RES 1 {self.nres}''')
            f.write('\nEND\nEND')
        return file_heat.split("/")[-1]

    def make_eq_input(self, rest, temp, total_ps, dt, nwrite):
        ''' Sets up the Amber file for the unrestrained equilibration
        '''

        nsteps_x = int(nwrite/dt) # How often to print trajectory
        rest_str = ""
        clean_prefix = self.f_prefix.split("/")[-1]
        def eq_file(steps, total_t, eqfile):
            with open(eqfile, 'w') as f:
                f.write(f"{clean_prefix}: {total_t}ps MD equilibration\n")
                f.write(" &cntrl\n")
                f.write(f'''  imin   = 0,
  irest  = 1,
  ntx    = 5,
  ntp    = 1,
  pres0  = 1.0, !Reference pressure
  taup   = 2.0, !Pressure relaxation time of 2ps
  ntc    = 2,     !SHAKE to constrain bonds with H
  cut    = {self.cutoff},  !non-bonded cutoff in Ams
  ntf    = 2, !Don't calculate forces for H-atoms
  ntt    = 3, gamma_ln = 5.0, !Langevin dynamics with collision frequency 5
  temp0  = {temp}, tempi  =  {temp}   !Reference (constant) temperature
  nstlim = {int(steps)},
  dt = {dt},
  ntpr = {nsteps_x}, !How often to write in out file
  ntwx = {nsteps_x}, !How often to write in traj file
  ntwr = {nsteps_x}, !How often to write in restart file
  iwrap  = 1,      !Coordinates written to restart & traj are "wrapped" into a primary box.
  ioutfm = 0,      !Formatted ASCII trajectory
  ntr    = 0,
  ig     = -1
/
Hold the DNA fixed with weak restrains
{int(rest)/2}
RES 1 {self.nres}''')
            f.write('\nEND\nEND')
                
            return eqfile
        
        # Base equilibration
        nsteps = total_ps / dt 
        file_eq = eq_file(nsteps, total_ps, self.f_prefix + "_eq2.in")
        #nsteps = total_ps[1] / dt 
        #file_eq2 = eq_file(nsteps, total_ps[1], self.f_prefix + "_eq-extra.in")
        return file_eq.split("/")[-1]

    def make_prod_input(self, edge_str, temp, total_ps, dt, nwrite):
        ''' Sets up the Amber file for the production run
            PENDING
        '''
        # Restrains
        ntr_i = 0
        if edge_str:
            ntr_i = 1

        nsteps = int(total_ps/dt)
        nsteps_x = int(nwrite/dt) # How often to print trajectory
        prodfile = self.f_prefix + "_prod.in"
        clean_prefix = self.f_prefix.split("/")[-1]
        rest_str = ""
        with open(prodfile, 'w') as f:
            f.write(f"{clean_prefix}: {total_ps/1000}ns MD production\n")
            f.write(" &cntrl\n")
            f.write(f'''  imin   = 0,
  irest  = 1, !Restart simulation form saved restart file
  ntx    = 5, !Coordinates and velocities read from a NetCDF file
  ntp    = 1,   !constant pressure (isotropic scaling)
  pres0  = 1.0, !Reference pressure
  taup   = 2.0, !Pressure relaxation time of 2ps
  ntc    = 2,     !SHAKE to constrain bonds with H
  cut    = {self.cutoff},  !non-bonded cutoff of 12A
  ntf    = 2, !Don't calculate forces for H-atoms
  ntt    = 3, gamma_ln = 5.0, !Langevin dynamics with collision frequency 5
  temp0  = {temp}, tempi  =  {temp}            !Reference temperature
  nstlim = {nsteps},  !Number of MD steps
  dt     = {dt},  !2fs time-step
  ntpr   = {nsteps_x},    !How often to write in out file: Every 10ps
  ntwx   = {nsteps_x},   !How often to write in mdcrd file
  ntwr   = {nsteps_x*10},   !How often to write in restrt file
  iwrap  = 1,      !Coordinates written to restart & traj are "wrapped" into a primary box.
  ioutfm = 0,      !Formatted ASCII trajectory
  ntr    = {ntr_i},{edge_str}
  ig     = -1,\n/\n''')

        return prodfile.split("/")[-1]
    
    def run_md(self, path, sampleFile, param, coord, min_cycles, eq_time, prod_time, dt, save_time,
               nodes, tasks, logfile, slurm_prefix):
        
        # Initialize bash script
        run_file = f"run_{sampleFile}.sh"
        f =  open(path+run_file, 'w')
        if slurm_prefix:
            f.write(slurm_prefix)
        else:
            f.write(f"#!/bin/sh\n#SBATCH --job-name={sampleFile}\n")
            f.write(f"#SBATCH --nodes={nodes}\n#SBATCH --ntasks={tasks}\n#SBATCH --output={logfile}\n\n")

        filemin1, filemin2 = self.make_min_input(edge_str=self.edgeRest, rest=self.min_rest, ncycles=min_cycles)

        ref = coord
        input1 = filemin1 
        summ1  = filemin1[:-2] + "out"
        coordi = coord
        coordf = sampleFile + "_min1.ncrst"
        

        f.write(f'{self.sander} -O -i {input1} -o {summ1} -p {param} '
                f'-c {coordi} -r {coordf} -ref {ref}\n'
                )
        
        ref = coordf
        input2 = filemin2
        summ2  = filemin2[:-2] + "out"
        coordi = coordf
        coordf = sampleFile + "_min2.ncrst"
                 
        f.write(f'{self.sander} -O -i {input2} -o {summ2} -p {param} '
                f'-c {coordi} -r {coordf} -ref {ref}\n'
                )
                
        fileheat = self.make_heat_input(tF=self.rtemp, 
                                        rest=self.eq_rest, total_ps=eq_time[0], dt=dt, twrite=0.2)
        fileeq   = self.make_eq_input(rest=self.eq_rest, temp=self.rtemp, 
                                        total_ps=eq_time[1], dt=dt, nwrite=save_time)
        
        input1 = fileheat
        summ1  = fileheat[:-2] + "out"
        coordi = coordf
        ref    = coordf
        coordf = sampleFile + "_eq1.ncrst"
        traj   = sampleFile + "_eq1.nc"

        f.write(f'{self.sander} -O -i {input1} -o {summ1} -p {param} '
                f'-c {coordi} -r {coordf} -x {traj} -ref {ref}\n'
                )
        
        input2 = fileeq
        summ2  = fileeq[:-2] + "out"
        coordi = coordf
        coordf = sampleFile + f"_eq2.ncrst"
        traj   = sampleFile + f"_eq2.nc"
        
        f.write(f'{self.sander} -O -i {input2} -o {summ2} -p {param} '
                f'-c {coordi} -r {coordf} -x {traj} -ref {ref}\n'
                )
        
        fileprod = self.make_prod_input(edge_str=self.edgeRest, temp=self.rtemp, 
                                        total_ps=prod_time, dt=dt, nwrite=save_time)
        input = fileprod
        summ  = fileprod[:-2] + "out"

        coordi = coordf
        coordf = sampleFile + "_prod.ncrst"
        traj   = sampleFile + "_prod.nc"

        f.write(f'{self.sander} -O -i {input} -o {summ} -p {param} '
                f'-c {coordi} -r {coordf} -x {traj} -ref {ref}\nwait\n'
                )
        f.close()
        
        jobID = subprocess.Popen(f"sbatch --parsable {run_file}", shell = True, cwd=path)
        return traj, coordf, jobID

    def run_extra_prod(self, path, sampleFile, param, coordf, prod_time, dt, save_time, iextra,
                       nodes, tasks, logfile, slurm_prefix):
        
        # Initialize bash script
        run_file = f"run_{sampleFile}_extra.sh"
        f =  open(path+run_file, 'w')
        if slurm_prefix:
            f.write(slurm_prefix)
        else:
            f.write(f"#!/bin/sh\n#SBATCH --job-name={sampleFile}\n")
            f.write(f"#SBATCH --nodes={nodes}\n#SBATCH --ntasks={tasks}\n#SBATCH --output={logfile}\n\n")

             
        fileprod = self.make_prod_input(edge_str=self.edgeRest, temp=self.rtemp, 
                                        total_ps=prod_time, dt=dt, nwrite=save_time)
        input_file = fileprod

        coordi = coordf
        ref = sampleFile + "_min2.ncrst"
        coordf = sampleFile + f"_prod_{iextra}.ncrst"
        traj   = sampleFile + f"_prod_{iextra}.nc"
        summ  = fileprod[:-3] + f"_{iextra}.out"

        f.write(f'{self.sander} -O -i {input_file} -o {summ} -p {param} '
                f'-c {coordi} -r {coordf} -x {traj} -ref {ref}\nwait\n'
                )
        f.close()
        
        jobID = subprocess.Popen(f"sbatch --parsable {run_file}", shell=True, cwd=path)
        return traj, coordf, jobID    
    
    def check_rmsd(self, param, trajs, step, rmsd_time, save_time, dt, save_path=""):
        nsteps_x = save_time / dt
        nlast = rmsd_time / dt
        
        # Calculate rmsd of last trajectory with cpptraj
        script_cont=f"parm {param}"
        for traj in trajs:
            script_cont += f"\ntrajin {traj}"
        script_cont += f"\nrms ToFirst :1-{self.nres}&!@H= first out rmsd_{step}.txt mass\nrun\n"
        f = open(save_path + "get_rmsd.in", 'w')
        f.write(script_cont)
        f.close()
        cp = subprocess.Popen(f"source {self.amber}/amber.sh", shell = True)
        cp.wait()
        subprocess.Popen(f"{self.amber}/bin/cpptraj -i get_rmsd.in", shell = True, cwd=save_path).wait()
        
        # Test equilibration of last 200ps of rmsd (traj was generated every nwrite ps)
        rmsd = np.loadtxt(f"{save_path}rmsd_{step}.txt")[:,1]
        avg = np.mean(rmsd)
        rmsd_last = rmsd[int(nlast/nsteps_x):]
        
        # Test how many times the data passes through the average en the last x ps 
        passes = np.where(np.diff(np.sign(rmsd_last-avg)))[0]
        if len(rmsd_last) < 1:
            print("The selected frame rmsd_time is longer than the available trajectory")
            rmsd_last = rmsd
            
        if rmsd_last[0] == avg: # when the start point is the avg, also count it
            np.insert(passes, 0, 0)
        if rmsd_last[-1] == avg: # when the last point is the avg, also count it
            np.append(passes, 0) 

        return len(passes), len(rmsd_last)


    def calculate_cofm(self, param, trajs, step, sel_string, cofm_time, save_time, dt, save_path):
        from commons.geom_utils import get_RAB
        
        nsteps_x = save_time / dt
        nlast = cofm_time / dt
        
        # get resid of dyes
        u = mda.Universe(param, trajs[0], format="TRJ")
        res_list = u.select_atoms("resname " + sel_string).residues
        resid1 = res_list[0].resid
        resid2 = res_list[-1].resid
        
        # Calculate center of mass distance
        rabs = []
        for traj in trajs:
            rab = get_RAB(param, traj[:-3], '', ntrajs=1, dt=1, resnum1=str(resid1), resnum2=str(resid2))
            rabs.append(rab)
        rabs = np.concatenate(rabs)
        np.savetxt(f"{save_path}cofm_{step}.txt", rabs, fmt='%.5f')
        
        avg = np.mean(rabs)
        rab_last = rabs[int(nlast/nsteps_x):]
        
        # Test how many times the data passes through the average en the last x ps 
        passes = np.where(np.diff(np.sign(rab_last-avg)))[0]
        if rab_last[0] == avg: # when the start point is the avg, also count it
            np.insert(passes, 0, 0)
        if rab_last[-1] == avg: # when the last point is the avg, also count it
            np.append(passes, 0) 
            
        print(resid1, resid2)
        
        return len(passes), len(rab_last)
        
        
def md_run(sample, path, amber_path, sample_frefix='dimer_', pdb=None, param=None, coord=None, 
           cutoff=12.0, edges_rest=None,
           min_cycles=[2000,2000], eq_runtime=[20,1000], prod_runtime=5000, dt=0.002,
           nodes=1, tasks=1, logfile="out.log", sander_path=None, slurm_prefix=None):
    """Running MD for the sample number <sample>: 2-step min, 2-step eq & production
        *** Note that this function is for running in a HPC SLURM environment

    Args:
        sample (int): Sample number to be run
        path (str): Path were sample files are saved
        amber_path (str): Path to Amber 
        sample_frefix (str, optional): Prefix with which sample files are saved. 
            Defaults to 'dimer'.
        pdb (str, optional): Path to PDB file if default of DNA+dye if default is not wanted.
        param (str, optional): Path to prmtop file if default of DNA+dye if default is not wanted.
        coord (str, optional): Path to rst7 file if default of DNA+dye if default is not wanted.
        cutoff (float, optional): Simulation cutoff distance. Defaults to 12.0.
        edges_rest (float, optional): Harmonic restrain force constant to be applied to DNA box edges. 
            If not given, no restrains will be applied. Units kcal/mol-A**2.
        min_cycles (list, optional): Num of cycles for [min1, min2]. Defaults to [2000,2000].
        eq_runtime (list, optional): Runtime of [heating, equilibration] in ps. Defaults to [20,1000].
        prod_runtime (int, optional): Production runtime in ps. Defaults to 4000.
        dt (float, optional): Time step in ps. Defaults to 0.002.
        nodes (int, optional): Num of nodes assigned to HPC run. Defaults to 1.
        tasks (int, optional): Num of tasks for HPC run. Defaults to 1.
        logfile (str, optional): Custom name for HPC run log file. Defaults to "out.log".
        sander_path(str,optional): The path to sander executable. 
                                  Defaults to non-MPI sander executable in <amber_path>/bin if not provided.
        slurm_prefix (str, optional): SLURM prefix if default is not desired. Defaults to None.
    """

    sampleF = sample_frefix + str(sample)
    if not param:
        param = sampleF + "_clean.prmtop"
    if not coord:
        coord = sampleF + "_clean.rst7"
    if not pdb:
        pdb = sampleF + "_clean.pdb"
    
    # Calculate number of non-solvent residues
    u = mda.Universe(path+pdb, format="PDB")
    nres = len(u.atoms.residues)

    # Find the edge residues
    from commons.geom_utils import find_term_res_pdb
    ter_res = []
    if edges_rest:
        ter_res = find_term_res_pdb(path+pdb, nres, start_res=1)
        print(ter_res)

    md = md_samples(amber_path, nres, sampleF, cutoff, sander_path, ter_res, edges_rest)

    savetime = 10
    trajf, coordf, jobID = md.run_md(path, sampleF, param.split("/")[-1], coord.split("/")[-1], 
                                     min_cycles, eq_runtime, prod_runtime, dt, savetime,
                                     nodes=nodes, tasks=tasks, logfile=logfile, slurm_prefix=slurm_prefix)    
    return #trajf, coordf, jobID


def rmsd_check(sample, path, amber_path, sample_frefix='dimer_', pdb=None, trajs=None, dye_res='CY3',
                  prod_runtime=5000, save_time=10, dt=0.002, rmsd_time=200, cutoff=12.0, edges_rest=None,
                  metric='RMSD', iextra=2, pass_fraction=10,
                  nodes=1, tasks=1, logfile='out.log', sander_path=None, slurm_prefix=None):
    """Checks if MD simulation converged. If not, will run additional production time. 

    Args:
        sample (int): Sample number to be run
        path (str): Path were sample files are saved
        amber_path (str): Path to Amber 
        sample_frefix (str, optional): Prefix with which sample files are saved. 
            Defaults to 'dimer'.
        pdb (str, optional): Path to PDB file if default of DNA+dye if default is not wanted.
        trajs (str, optional): List of .nc amber trajectories for calulcation. 
            If not given, default is used.
        dye_res (str, optional): Name of dye residue in PDB and amber files. Defaults to 'CY3'.
        prod_runtime (int, optional): Runtime for extra production. Defaults to 5000.
        save_time (int, optional): time step for metric calculation. Defaults to 10.
        dt (float, optional): Time step in ps. Defaults to 0.002.
        rmsd_time (int, optional): time in ps to calculate the metric. Defaults to 200.
        cutoff (float, optional): Simulation cutoff distance. Defaults to 12.0.
        edges_rest (float, optional): Harmonic restrain force constant to be applied to DNA box edges. 
            If not given, no restrains will be applied. Units kcal/mol-A**2.
        metric (str, optional): Metric to asses convergence: RMSD or dimer CofMass dist. Defaults to 'RMSD'.
        iextra (int, optional): index for additional prod runs. Defaults to 2.
        pass_fraction (int, optional): Min percent of times data passes through avg. 
            If total passes are below this number, extra production is run. Defaults to 10.
        nodes (int, optional): Num of nodes assigned to HPC run. Defaults to 1.
        tasks (int, optional): Num of tasks for HPC run. Defaults to 1.
        logfile (str, optional): Custom name for HPC run log file. Defaults to "out.log".
        sander_path(str,optional): The path to sander executable. 
                                  Defaults to non-MPI sander executable in <amber_path>/bin if not provided.
        slurm_prefix (str, optional): SLURM prefix if default is not desired. Defaults to None.

    Raises:
        NotImplementedError: Wrong type of metric used

    Returns:
        int: Numer of passes through mean
    """
    
    sampleF = sample_frefix + str(sample)
    if not trajs:
        traj1 = sampleF + "_eq2.nc"
        traj2 = sampleF + "_prod.nc"
        trajs = [traj1, traj2]
    param = sampleF + "_clean.prmtop"
    if not pdb:
        pdb = sampleF + "_clean.pdb"
        
    coordf = trajs[-1]+"rst"  
    
    # Calculate number of non-solvent residues
    u = mda.Universe(pdb, format="PDB")
    nres = len(u.atoms.residues)
    
    # Find the edge residues
    from commons.geom_utils import find_term_res_pdb
    ter_res = []
    if edges_rest:
        ter_res = find_term_res_pdb(path+pdb,dist_min=6)
        print(ter_res)

    md = md_samples(amber_path, nres, sampleF, cutoff, sander_path, ter_res, edges_rest)
    if metric == "RMSD":
        passes, total = md.check_rmsd(param, trajs, sample, rmsd_time, save_time, dt, save_path=path)
    elif metric == "CofM":
        passes, total = md.calculate_cofm(param, trajs, sample, dye_res, rmsd_time, save_time, dt, save_path=path)
    else:
        raise NotImplementedError("Only RMSD and CofM metrics are implemented")
        
    pass_min = pass_fraction/100 * total
    print(f'Numer of Passes through the avg {passes} ({passes*100/total}), from total {total}, and min metric {pass_min}')
    if passes < pass_min: # The data didn't passes enough times through the average
        # Run extra production if simulation not yet converged
        trajf, coordf, jobID = md.run_extra_prod(path, sampleF, param.split("/")[-1], coordf.split("/")[-1],
                                                 prod_runtime, dt, save_time, iextra=iextra, 
                                                 nodes=nodes, tasks=tasks, logfile=logfile, slurm_prefix=slurm_prefix)
    
    return passes

