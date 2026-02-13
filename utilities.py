import numpy as np
import pandas as pd
import io
import os
import itk


def printdt(*args):
    import datetime
    print(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), *args)


def get_schneider_density(hu_value):
    """
    Interpolates density from HU based on Schneider 2000 table.
    hu_points and density_points are taken directly from Schneider2000DensitiesTable.txt.
    """
    hu_points = np.array([-1000, -98, -97, 14, 23, 100, 101, 1600, 3000])
    density_points = np.array([0.00121, 0.93, 0.930486, 1.03, 1.031, 1.1199, 1.0762, 1.9642, 2.8])
    print(hu_value)
    # np.interp handles values outside the range by using the edge values
    return float(np.interp(hu_value, hu_points, density_points))


def get_schneider_materials(output_path=None):
    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    with open("Schneider2000MaterialsTable.txt") as f:
        mat = f.read()
    data = mat.split("=")[-1]
    names = mat.split("#")[-2].replace('\n', '').split(' ')
    names = [x for x in names if x]
    names.append('Material')
    df = pd.read_csv(io.StringIO(data), sep=r'\s+', names=names)
    df.drop_duplicates('Material', inplace=True)
    df.set_index('Material', inplace=True)
    df['HU'] = df['HU'].astype(int)
    float_cols = [col for col in df.columns if col not in ['HU', 'Material']]
    df[float_cols] = df[float_cols].astype(float)
    current_output_file = "Schneider_HU_mapping.csv" if output_path is None else os.path.join(output_path, "Schneider_HU_mapping.csv")
    df.to_csv(current_output_file)

    element_map = {
        "Hydrogen": "H", "Carbon": "C", "Nitrogen": "N", "Oxygen": "O",
        "Sodium": "Na", "Magnesium": "Mg", "Phosphor": "P", "Sulfur": "S",
        "Chlorine": "Cl", "Argon": "Ar", "Potassium": "K", "Calcium": "Ca"
    }

    current_output_file = "materials.txt" if output_path is None else os.path.join(output_path, "materials.txt")
    with open(current_output_file, "w") as f:
        f.write("# Generated GGEMS Material File\n\n")
        for mat in df.index.values:
            hu = df.loc[mat, 'HU']
            density = get_schneider_density(hu)
            f.write(f"[Material: {mat}]\n")
            f.write(f"density = {density:.5f}\n")
            elemental_composition_str = ""
            for col in float_cols:
                short_name = element_map.get(col, col)  # Use map, or default to original
                if col != float_cols[-1]:
                    elemental_composition_str += f"{short_name} = {df.loc[mat, col]}; "
                else:
                    elemental_composition_str += f"{short_name} = {df.loc[mat, col]}\n\n"
            f.write(elemental_composition_str)


def get_ct_materials(path_to_ct, output_path, materials_file=None):
    ct = itk.imread(path_to_ct)
    ct_array = itk.array_from_image(ct)
    ct_labels = np.zeros_like(ct_array, dtype=np.float32)

    ct_sep_string = ("################################################################################\n"
                     "#                            MATERIALS FROM CT DATA                            #\n"
                     "################################################################################")
    if materials_file is None:
        materials_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "ggems_materials.txt")
    with open(materials_file) as f:
        materials = f.read()

    materials = materials.split(ct_sep_string)[-1]
    materials = materials.split("# ")
    materials = [mat.replace('\n\n', '') for mat in materials if mat.startswith('Material')]
    materials_dict = {}
    range_file = os.path.join(output_path, path_to_ct.split(os.sep)[-1].split('.')[0] + '_range_phantom.txt')
    open(range_file, 'w').close()
    for material in materials:
        lines = material.split('\n')
        label = int(lines[0].split(' ')[1])
        hu_lower_threshold = float(lines[0].split('H=[ ')[-1].split(';')[0])
        hu_upper_threshold = float(lines[0].split('H=[ ')[-1].split(';')[-1].replace(' ]', ''))
        name = lines[1].split(':')[0]
        materials_dict[name] = label
        with open(range_file, 'a') as f:
            f.write(f"{label} {label} {name}\n")

        mask = (ct_array > hu_lower_threshold) & (ct_array <= hu_upper_threshold)
        ct_labels[mask] = label

    ct_labels = itk.image_from_array(ct_labels)
    ct_labels.CopyInformation(ct)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = path_to_ct.split(os.sep)[-1]
    path_to_output_file = os.path.join(output_path, output_file)
    itk.imwrite(ct_labels, path_to_output_file)

    return materials_dict


# Function to resample an itk Image given a reference itk Image
def resample_to_reference(volume, reference, interpolation_mode="bspline", ct=False):

    if interpolation_mode == "nearestneighbour":  # for masks
        interpolator = itk.NearestNeighborInterpolateImageFunction
    elif interpolation_mode == "linear":
        interpolator = itk.LinearInterpolateImageFunction
    else:
        interpolator = itk.BSplineInterpolateImageFunction

    if isinstance(volume, str):
        volume = itk.imread(volume)

    if isinstance(reference, str):
        reference = itk.imread(reference)

    out_of_fov_fill = 0 if not ct else -1000
    return itk.resample_image_filter(
        volume,
        interpolator=interpolator.New(volume),
        use_reference_image=True,
        reference_image=reference,
        default_pixel_value=out_of_fov_fill
    )


def generate_lu177_spectrum(output_file="resources/Lu177_Beta_Spectrum.dat"):
    # Energy bins in MeV (from 0 to max E_max of 0.498 MeV)
    energies = np.linspace(0.001, 0.498, 100)

    # Beta branch data: (E_max in MeV, Intensity fraction)
    branches = [
        (0.4968, 0.7944),   # Main branch
        (0.3839, 0.0889),   # Middle branch
        (0.1755, 0.1166)    # Low branch
    ]

    total_spectrum = np.zeros_like(energies)

    for e_max, intensity in branches:
        # Simplified Fermi distribution approximation for allowed beta transitions: P(E) ~ sqrt(E) * (E_max - E)^2
        mask = energies < e_max
        prob = np.zeros_like(energies)
        prob[mask] = np.sqrt(energies[mask]) * (e_max - energies[mask]) ** 2

        # Normalize branch and scale by intensity
        if prob.sum() > 0:
            total_spectrum += (prob / prob.sum()) * intensity

    # Save to file in GGEMS format (Energy Probability)
    with open(output_file, 'w') as f:
        f.write("# Lu-177 Combined Beta Spectrum\n")
        f.write("# Energy(MeV) Probability\n")
        for e, p in zip(energies, total_spectrum):
            f.write(f"{e:.6f} {p:.6e}\n")

    print(f"✅ Spectrum file generated at: {output_file}")


def get_materials_densities(output_path = None, materials_file=None):


    ct_sep_string = ("################################################################################\n"
                     "#                            MATERIALS FROM CT DATA                            #\n"
                     "################################################################################")
    if materials_file is None:
        materials_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "ggems_materials.txt")
    with open(materials_file) as f:
        materials = f.read()

    materials = materials.split(ct_sep_string)[-1]
    materials = materials.split("# ")
    materials = [mat.replace('\n\n', '') for mat in materials if mat.startswith('Material')]
    densities_dict = {
        "Material": [],
        "Label": [],
        "Density[mg/cm3]": [],
        "LowerThreshold[HU]": [],
        "UpperThreshold[HU]": []
    }
    for material in materials:
        lines = material.split('\n')

        name = lines[1].split(':')[0]
        hu_lower_threshold = float(lines[0].split('H=[ ')[-1].split(';')[0])
        hu_upper_threshold = float(lines[0].split('H=[ ')[-1].split(';')[-1].replace(' ]', ''))

        label = int(lines[0].split(' ')[1])
        density = float(lines[1].split(' ')[1].replace('d=', ''))

        density_unit = lines[1].split(' ')[2]
        if density_unit == 'g/cm3':
            density = density * 1000

        densities_dict['Material'].append(name)
        densities_dict['Label'].append(label)
        densities_dict['Density[mg/cm3]'].append(density)
        densities_dict['LowerThreshold[HU]'].append(hu_lower_threshold)
        densities_dict['UpperThreshold[HU]'].append(hu_upper_threshold)

    df = pd.DataFrame(densities_dict)
    if output_path is None:
        output_path = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(output_path, "densities.csv"), index=False)

    return densities_dict

