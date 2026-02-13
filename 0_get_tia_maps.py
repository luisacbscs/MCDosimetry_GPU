import itk
import os
import numpy as np
from skimage import morphology
import pandas as pd
from utilities import *
from tqdm import tqdm
import sys
import torch
itk.ProcessObject.SetGlobalWarningDisplay(False)

path = r"D:/Pluvicto"

# output paths
for folder in ["TIA_mask", "TIA", "TIA_CT"]:
    if not os.path.exists(os.path.join(path, folder)):
        os.makedirs(os.path.join(path, folder))

timepoints_df = pd.read_csv(os.path.join(path, "timepoints.csv"))
for col in ["T0", "T1", "T2", "T3"]:
    timepoints_df[col] = pd.to_datetime(timepoints_df[col])
timepoints_df.dropna(subset='T0', inplace=True)

#for index, row in tqdm(timepoints_df.iterrows(), total=len(timepoints_df), file=sys.stdout, desc="getting TIA maps..."):
for index, row in timepoints_df.iterrows():
    patient_id = row["NewID"]
    cycle = row["Cycle"]
    print("="* 64, patient_id, cycle)

    data = {
        "T1": f"{patient_id}_SPECT_{cycle}-T1.nii.gz",
        "T2": f"{patient_id}_SPECT_{cycle}-T2.nii.gz",
        "T3": f"{patient_id}_SPECT_{cycle}-T3.nii.gz",
    }

    adm_dtime = timepoints_df[(timepoints_df["NewID"] == patient_id) & (timepoints_df["Cycle"] == cycle)]["T0"].values[0]
    acquisition_times = []
    timepoints = []
    ct_paths = []
    for T in data.keys():
        img = itk.imread(os.path.join(path, "SPECT", data[T])) \
            if os.path.exists(os.path.join(path, "SPECT", f"{patient_id}_SPECT_{cycle}-{T}.nii.gz")) else None
        if img is not None:
            timepoints.append(img)
            t_dtime = timepoints_df[(timepoints_df["NewID"] == patient_id) & (timepoints_df["Cycle"] == cycle)][T].values[0]
            acquisition_times.append(pd.to_timedelta(t_dtime - adm_dtime).total_seconds() / 3600)
            ct_paths.append(os.path.join(path, "SPECT_CT", f"{patient_id}_CT_{cycle}-{T}.nii.gz"))
    if len(timepoints) < 2:
        continue

    reference_timepoint = timepoints[0]
    reference_arr = np.asarray(reference_timepoint)
    # saving the first available timepoint's CT in .mhd
    ct = itk.imread(ct_paths[0])
    ct = resample_to_reference(ct, reference_timepoint)
    itk.imwrite(ct, os.path.join(path, "TIA_CT", f'{patient_id}_{cycle}_CT.mhd'))

    # getting mask of foreground
    threshold = np.max(reference_arr) * 0.005
    mask_arr = np.zeros_like(reference_arr)
    mask_arr[reference_arr > threshold] = 1
    selem = morphology.ball(2)
    mask_arr = morphology.closing(mask_arr, footprint=selem).astype(np.uint8)
    mask = itk.image_from_array(mask_arr)
    mask.CopyInformation(reference_timepoint)
    itk.imwrite(mask, os.path.join(path, "TIA_mask", f"{patient_id}_SPECT_{cycle}_mask.nii.gz"))

    device = torch.device("cuda")
    l_phys = torch.tensor(np.log(2) / (6.647 * 24), device=device)

    gpu_tensors = [torch.from_numpy(itk.array_from_image(timepoints[0])).to(device).float()]
    # moving to GPU
    for i in range(1, len(timepoints)):
        resampled = resample_to_reference(timepoints[i], reference_timepoint)
        tensor = torch.from_numpy(itk.array_from_image(resampled)).to(device).float()
        gpu_tensors.append(tensor)

    print(f"Number of GPU tensors: {len(gpu_tensors)}")
    print(f"Acquisition Times: {acquisition_times}")

    data_stack_gpu = torch.stack(gpu_tensors)  # Shape: (Timepoints, Z, Y, X)
    mask_gpu = torch.from_numpy(mask_arr).to(device).bool()
    t = torch.tensor(acquisition_times, device=device).float()
    l_phys = torch.tensor(np.log(2) / (6.647 * 24), device=device)

    a1 = data_stack_gpu[0]
    t1 = t[0]
    area_uptake = 0.5 * a1 * t1  # Linear uptake from 0 to first scan

    if len(data_stack_gpu) == 3:    # all timepoints available
        a2, a3 = data_stack_gpu[1], data_stack_gpu[2]
        t2, t3 = t[1], t[2]

        # Segment A: Trapezoidal area between first and second scan
        area_measured = 0.5 * (a1 + a2) * (t2 - t1)

        # Segment B: Exponential tail from second scan onwards
        # We use a3/a2 to find the washout slope
        a2_safe = torch.where(a2 > 1e-4, a2, torch.tensor(1e-4, device=device))
        ratio = torch.clamp(a3 / a2_safe, max=0.999)
        l_eff = torch.clamp(-torch.log(ratio) / (t3 - t2), min=l_phys)

        area_tail = a2 / l_eff  # Integration from t2 to infinity
        area_main = area_measured + area_tail

    else:   # one timepoint missing
        a2 = data_stack_gpu[1]
        t2 = t[1]

        a1_safe = torch.where(a1 > 1e-4, a1, torch.tensor(1e-4, device=device))
        ratio = torch.clamp(a2 / a1_safe, max=0.999)
        l_eff = torch.clamp(-torch.log(ratio) / (t2 - t1), min=l_phys)

        area_main = a1 / l_eff  # Integration from t1 to infinity

    combined_tia = area_uptake + area_main
    combined_tia = torch.nan_to_num(combined_tia, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Max TIA before masking: {torch.max(combined_tia).item()}")
    tia_map_gpu = torch.where(mask_gpu, combined_tia, torch.tensor(0.0, device=device))
    print(f"Max TIA after masking: {torch.max(tia_map_gpu).item()}")

    voxel_volume = reference_timepoint['spacing'][0] * reference_timepoint['spacing'][1] * \
                   reference_timepoint['spacing'][2]
    voxel_volume = voxel_volume / 1000  # mm3 --> cm3 (mL)
    tia_map_gpu = tia_map_gpu * voxel_volume * 3600  # (Activity / mL) . hours --> Bq . s

    # moving back to CPU
    tia_map = tia_map_gpu.cpu().numpy()

    tia_itk = itk.image_from_array(tia_map)
    tia_itk.CopyInformation(reference_timepoint)
    itk.imwrite(tia_itk, os.path.join(path, "TIA", f'{patient_id}_{cycle}_TIA_Map.mhd'))
    torch.cuda.empty_cache()
