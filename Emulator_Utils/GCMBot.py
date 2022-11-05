import subprocess
import psutil
import shutil
import os

# Press the green button in the gutter to run the script.
import sys
import time

limit_num_runs = 10
variable_name = "tautot"


def replace_func(f):
    return f"""!  GCM input file\n\
!  original 4-27-05\n\
!\n\
! runnumx - (real) run identifier\n\
! dlat    - (real) degrees between latitude grid points\n\
! jm      - (integer) number of latitude grid points\n\
! im      - (integer) number of longitude grid points\n\
! nlay    - (integer) number of layers\n\
! psf     - (real) average surface pressure (mbar)\n\
! ptrop   - (real) pressure at the tropopause (mbar)\n\
! dtm     - (real) requested time step (minutes)\n\
! tautot  - (real) visible dust optical depth at the reference pressure level\n\
! rptau   - (real) the reference pressure level tautot uses (mbar)\n\
! taue    - (real) requested run time (hours)\n\
! tauh    - (real) history output every tauh hours (hours)\n\
! tauid   - (real) starting time in days - leave 0 for now\n\
! tauih   - (real) starting time of run (hours); 0 for cold starts, equal\n\
!                  to the time of the first record of a warm start file.\n\
! nc3     - (integer) a full pass through COMP3 is done every NC3 time\n\
!                     steps\n\
!\n\
! rsetsw  - (integer) cold start/ warm start flag; = 1 for cold starts\n\
!                                                  = 0 for warm starts\n\
! lday    - (integer) day of a Mars year corresponding to a given Ls.\n\
!                     some values:    Ls       lday\n\
!                                   ------    ------\n\
!                                      0       173\n\
!                                     90       366\n\
!                                    180       545\n\
!                                    270        19\n\
!\n\
! conrnu  - (real) dust mixing ratio scale height\n\
!                  Standard value = 0.03   ~25km half-height\n\
!                  CONRNU = 0.003          ~50km half-height\n\
!                  CONRNU = 0.5            ~10km half-height\n\
!\n\
!-----------------------------------------------------------------------------#\n\
!                       \n\
! read in input.f \n\
! rsetsw  -  1 for cold starts\n\
!            0 for warm starts\n\
! do not change "&inputnl"\n\
&inputnl \n\
  runnumx = 2014.11\n\
  dlat = 5.0     jm = 36    im = 60   nlay = 24\n\
  psf  = 7.010   ptrop = 0.0008\n\
  dtm  = 2.0     nc3  =  8\n\
  tautot = {f}  rptau = 6.1  conrnu = 0.03\n\
  taue = 481.5  tauh = 1.5    tauid = 0.0  tauih = 0.0\n\
  rsetsw = 1\n\
  cloudon = .false.\n\
  active_dust = .true.\n\
  active_water = .true.\n\
  microphysics = .true.\n\
  co2scav = .true.\n\
  timesplit = .false.\n\
  albfeed = .false.\n\
  latent_heat = .false.\n\
  vdust = .true.\n\
  icealb = 0.4  icethresh_depth = 5.0\n\
  dtsplit = 30.0\n\
  vary_conr = .false.\n\
 /\n\
\n\
! read in insdet.f \n\
! do not change "&insdetnl"\n\
&insdetnl \n\
  lday = 366 /\n\
\n\
"""


if __name__ == '__main__':

    for run in range(101):  # This runs from tautot 0 through 10 stepping by .1

        while len([x.name() for x in psutil.process_iter() if "gcm2.3" in x.name()]) >= limit_num_runs:
            for process in psutil.process_iter():
                # print(process.status(), process.pid)
                if process.status() == "zombie" and process.ppid() == os.getpid():
                    os.waitpid(process.pid, 0)

            time.sleep(1)
        else:
            current_run_var = str(float(run) / 10).zfill(1)
            dir_name = f"dir_{variable_name}_MY24_{current_run_var}"
            os.makedirs(f"/home/user/GCM/output/{dir_name}", exist_ok=True)
            os.chdir(f"/home/user/GCM/output/{dir_name}")
            print(f"current run {current_run_var} -- current dir {os.getcwd()}")
            with open(f"./mars2", 'w') as out:
                out.write(replace_func(current_run_var))

            dest = shutil.copytree("/home/user/GCM/data/", f"./data/")
            shutil.copy("/home/user/GCM/code/gcm2.3", f"../")
            # os.rename("/home/user/GCM/code/gcm2.3", f"/home/user/GCM/code/gcm2.3_{dir_name}")
            cp = subprocess.Popen(["nohup", f"./gcm2.3"], stdin=open("./mars2"), stdout=open("./m.out", 'w'),
                                  stderr=sys.stderr, preexec_fn=os.setpgrp)

