import os
import glob
import subprocess
import time
venv_python = r"C:\dev\mcdosimetry_gpu\ggems-env\Scripts\python.exe"

tia_dir = r"D:\Pluvicto\TIA"
# Get all patient IDs
tia_files = glob.glob(os.path.join(tia_dir, "*_TIA_Map.mhd"))
patient_ids = [os.path.basename(f).replace("_TIA_Map.mhd", "") for f in tia_files]

for id_ in patient_ids:

    print("="*128, id_)

    # Gamma
    gamma = False
    if os.path.isfile(os.path.join(r"D:\Pluvicto\DoseMaps", id_, "Gamma_edep.mhd")):
        print(f"Gamma simulation for {id_} already done ✅")
        gamma = True
    else:
        gamma_process = subprocess.run([venv_python, "gamma_simulation_worker.py", id_])

        if gamma_process.returncode != 0:
            print(f"‼️ Process for {id_} failed with exit code {gamma_process.returncode}")
            print("Retrying...")
            subprocess.run([venv_python, "gamma_simulation_worker.py", id_])

    # Beta
    beta = False
    if os.path.isfile(os.path.join(r"D:\Pluvicto\DoseMaps", id_, "Beta_edep.mhd")):
        print(f"Beta simulation for {id_} already done ✅")
        beta = True
    else:
        beta_process = subprocess.run([venv_python, "beta_simulation_worker.py", id_])

        if beta_process.returncode != 0:
            print(f"‼️ Process for {id_} failed with exit code {beta_process.returncode}")
            print("Retrying...")
            subprocess.run([venv_python, "beta_simulation_worker.py", id_])

    # The "Cool Down" still helps Blackwell hardware stability
    if not beta or not gamma:
        print("Waiting 7 seconds for hardware res(e)t :)")
        time.sleep(7)
