import os
import itk
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utilities import *


def get_hu_to_density_curve(densities_csv_path):
    """Creates a continuous HU-to-density (kg/m3) interpolation function."""
    df = pd.read_csv(densities_csv_path)
    hu_midpoints = (df['LowerThreshold[HU]'] + df['UpperThreshold[HU]']) / 2.0
    densities = df['Density[mg/cm3]'].values  # Already normalized to kg/m3
    return interp1d(hu_midpoints, densities, kind='linear', fill_value="extrapolate")


def generate_clinical_dose(path_beta_edep, path_gamma_edep, path_tia, path_ct, n_sim_beta=1e9, n_sim_gamma=1e9):
    # 1. Load Images
    beta_itk = itk.imread(path_beta_edep)
    gamma_itk = itk.imread(path_gamma_edep)
    tia_itk = itk.imread(path_tia)
    ct_itk = itk.imread(path_ct)

    # ALIGNMENT
    beta_itk.SetOrigin(tia_itk.GetOrigin())
    beta_itk.SetDirection(tia_itk.GetDirection())
    gamma_itk.SetOrigin(tia_itk.GetOrigin())
    gamma_itk.SetDirection(tia_itk.GetDirection())

    tia_resampled = resample_to_reference(tia_itk, reference=beta_itk, interpolation_mode="linear")
    ct_resampled = resample_to_reference(ct_itk, reference=beta_itk, interpolation_mode="linear", ct=True)

    beta_raw = itk.array_from_image(beta_itk)
    gamma_raw = itk.array_from_image(gamma_itk)
    tia_vol = itk.array_from_image(tia_resampled)
    ct_hu = itk.array_from_image(ct_resampled)

    total_decays = np.sum(tia_vol)
    print(f"Total Physical Decays (Bq.s): {total_decays:.2e}")

    # 2. Continuous Density Mapping (kg/m3)
    fit_function = get_hu_to_density_curve("densities.csv")
    density_map = fit_function(ct_hu)

    # 3. REPLICATE GGEMS MINIMUM DENSITY CUTOFF
    # 0.1 g/cm3 = 100 kg/m3 (this removes air/noise artefacts)
    min_density_kg_m3 = 100.0
    density_map_clipped = np.maximum(density_map, min_density_kg_m3)

    # 4. Calculate Voxel Mass (kg) using clipped density
    spacing = np.array(tia_itk.GetSpacing())
    voxel_vol_m3 = np.prod(spacing) * 1e-9  # mm3 --> m3
    voxel_mass_map = density_map_clipped * voxel_vol_m3

    # 5. Physical Constants and Yields
    mev_to_joules = 1.60218e-13
    total_physical_decays = np.sum(tia_vol)
    #gamma_yield = 0.1041 + 0.0623 --> not necessary - already taken into consideration when writing the SPECT images

    # 6. Dose Calculation
    def calculate_gy(raw_data, n_sim, yield_factor):
        scaling = (total_physical_decays / n_sim) * yield_factor * mev_to_joules
        # Dividing by the clipped mass map prevents infinite/massive values
        return (raw_data * scaling) / voxel_mass_map

    beta_gy = calculate_gy(beta_raw, n_sim_beta, 1.0)
    gamma_gy = calculate_gy(gamma_raw, n_sim_gamma, 1.0)

    # 7. Final Result
    total_dose_gy = beta_gy + gamma_gy
    final_img = itk.image_from_array(total_dose_gy)
    final_img.CopyInformation(tia_itk)
    return final_img


# Execution
path = r"D:\Pluvicto"
for id_ in sorted(os.listdir(os.path.join(path, "DoseMaps"))):

    if not os.path.isfile(os.path.join(path, 'DoseMaps', id_, "Beta_edep.mhd")):
        print(f"‼️ Missing beta simulation for {id_}")
        continue
    if not os.path.isfile(os.path.join(path, 'DoseMaps', id_, "Gamma_edep.mhd")):
        print(f"‼️ Missing gamma simulation for {id_}")
        continue

    total_gy_img = generate_clinical_dose(
        os.path.join(path, 'DoseMaps', id_, "Beta_edep.mhd"),
        os.path.join(path, 'DoseMaps', id_, "Gamma_edep.mhd"),
        os.path.join(path, "TIA", f"{id_}_TIA_Map.mhd"),
        os.path.join(path, "TIA_CT", f"{id_}_CT.mhd"),
        n_sim_beta=1e9,
        n_sim_gamma=1e9
    )

    itk.imwrite(total_gy_img, os.path.join(path, "DoseMaps", id_, "dose_Gy.mhd"))
    print(f"{id_} ✅ (dose map generated)")
