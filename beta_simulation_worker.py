import sys
import os
from ggems import *
from utilities import *
import shutil


def run_beta_simulation(id_):

    tia_file = f"D:\\Pluvicto\\TIA\\{id_}_TIA_Map.mhd"
    ct_file = f"D:\\Pluvicto\\TIA_CT\\{id_}_CT.mhd"
    out_path = f"D:\\Pluvicto\\DoseMaps\\{id_}"
    os.makedirs(out_path, exist_ok=True)

    print("[Beta]", tia_file, "x", ct_file, "-->", out_path)

    path_to_materials = os.path.join(os.path.dirname(os.path.abspath(__file__)), "materials.txt")
    if not os.path.isfile(path_to_materials):
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "ggems_materials.txt"),
                    path_to_materials)
    materials_database_manager = GGEMSMaterialsDatabaseManager()
    materials_database_manager.set_materials(path_to_materials)

    materials_path = os.path.dirname(ct_file) + '_GGEMS_Materials'
    if not os.path.exists(materials_path):
        os.makedirs(materials_path)
    materials = get_ct_materials(ct_file, output_path=materials_path, materials_file=path_to_materials)
    ct_materials_file = os.path.join(materials_path, ct_file.split(os.sep)[-1])
    ct_range_file = os.path.join(materials_path, ct_file.split(os.sep)[-1].split('.')[0] + '_range_phantom.txt')

    ct = itk.imread(ct_materials_file)
    spacing = ct.GetSpacing()  # (Sx, Sy, Sz) in mm
    size = itk.size(ct)  # (Nx, Ny, Nz) in voxels
    sx, sy, sz = spacing

    opencl_manager = GGEMSOpenCLManager()
    opencl_manager.set_device_to_activate('gpu', 'nvidia')

    processes_manager = GGEMSProcessesManager()
    processes_manager.add_process('Ionisation', 'e-', 'all')
    processes_manager.add_process('Bremsstrahlung', 'e-', 'all')
    processes_manager.add_process('MultipleScattering', 'e-', 'all')

    range_cuts_manager = GGEMSRangeCutsManager()
    range_cuts_manager.set_cut('e-', 0.1, 'mm', 'all')

    phantom = GGEMSVoxelizedPhantom('phantom')
    phantom.set_phantom(ct_materials_file, ct_range_file)
    phantom.set_rotation(0.0, 0.0, 0.0, 'deg')
    phantom.set_position(0.0, 0.0, 0.0, 'mm')
    phantom.set_visible(True)

    dosimetry = GGEMSDosimetryCalculator()
    dosimetry.attach_to_navigator("phantom")
    dosimetry.set_output_basename(os.path.join(out_path, "Beta"))
    dosimetry.set_dosel_size(sx, sy, sz, 'mm')
    dosimetry.water_reference(False)
    dosimetry.minimum_density(0.1, 'g/cm3')
    dosimetry.set_tle(True)  # Fast Track Length Estimator

    dosimetry.uncertainty(True)
    dosimetry.photon_tracking(True)
    dosimetry.edep(True)
    dosimetry.hit(True)
    dosimetry.edep_squared(True)

    voxel_source = GGEMSVoxelizedSource('voxel_source')
    voxel_source.set_phantom_source(tia_file)
    number_of_particles = int(1e9)
    voxel_source.set_number_of_particles(number_of_particles)
    voxel_source.set_position(0.0, 0.0, 0.0, 'mm')

    voxel_source.set_source_particle_type('e-')
    voxel_source.set_polyenergy('resources/Lu177_Beta_Spectrum.dat')

    # Run
    ggems = GGEMS()
    ggems.opencl_verbose(False)
    ggems.material_database_verbose(False)
    ggems.navigator_verbose(False)
    ggems.source_verbose(False)
    ggems.memory_verbose(False)
    ggems.process_verbose(False)
    ggems.range_cuts_verbose(False)
    ggems.random_verbose(False)
    ggems.profiling_verbose(False)
    ggems.tracking_verbose(False, 0)
    seed = 777
    ggems.initialize(seed)
    ggems.run()

    dosimetry.delete()
    ggems.delete()


if __name__ == "__main__":
    patient_id = sys.argv[1]
    run_beta_simulation(patient_id)
