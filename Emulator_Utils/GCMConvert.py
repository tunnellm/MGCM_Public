import subprocess
import psutil
import shutil
import os
import time

variable_name = "tautot"
base_path = "/home/user/GCM/output/"

if __name__ == "__main__":

    for run in [10, 100]:

            file_name = f"dir_{variable_name}_MY24_{str(float(run) / 10).zfill(1)}"
            cp = subprocess.Popen(["MarsFiles.py", f"/home/user/GCM/output/{file_name}/fort.11", "-fv3", "daily"])
            cp.wait(250)
            cp = subprocess.Popen(["MarsFiles.py", f"/home/user/GCM/output/{file_name}/fort.11_0002", "-fv3", "daily"])
            cp.wait(250)
