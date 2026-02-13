import sys
import os
from ggems import *
from utilities import *
import shutil

tia_file = r"D:\Pluvicto\TIA\PLV0001_C1_TIA_Map.mhd"

ct_file = r"D:\Pluvicto\TIA_CT\PLV0001_C1_CT.mhd"

out_path = r"D:\Pluvicto\DoseMaps"

id_ = tia_file.split(os.sep)[-1].replace("_TIA_Map", "").replace(".mhd", "")
current_output_path = os.path.join(out_path, id_)
if not os.path.exists(current_output_path):
    os.makedirs(current_output_path)


# 1. PRE-PROCESSING: Generate materials.txt and Label Map
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
nx, ny, nz = size

# 2. HARDWARE SETUP: Target the Blackwell GPU
opencl_manager = GGEMSOpenCLManager()
#print(dir(opencl_manager)) # This will list every method available to you
# Forces simulation to bypass Intel iGPU and use NVIDIA
opencl_manager.set_device_to_activate('gpu', 'nvidia')
opencl_manager.print_infos()

# 3. GEOMETRY: Load the Voxelized Anatomy
phantom = GGEMSVoxelizedPhantom('phantom')
phantom.set_phantom(ct_materials_file, ct_range_file)
phantom.set_rotation(0.0, 0.0, 0.0, 'deg')
phantom.set_position(0.0, 0.0, 0.0, 'mm')
phantom.set_visible(True)

# 4. DOSIMETRY: Score absorbed dose
dosimetry = GGEMSDosimetryCalculator()
dosimetry.attach_to_navigator("phantom")
dosimetry.set_output_basename(os.path.join(current_output_path, id_))
dosimetry.set_dosel_size(sx, sy, sz, 'mm')
dosimetry.water_reference(False)
dosimetry.minimum_density(0.1, 'g/cm3')
dosimetry.set_tle(True)  # Fast Track Length Estimator

dosimetry.uncertainty(True)
dosimetry.photon_tracking(True)
dosimetry.edep(True)
dosimetry.hit(True)
dosimetry.edep_squared(True)

# 5. PHYSICS
processes_manager = GGEMSProcessesManager()
processes_manager.add_process('Compton', 'gamma', 'all')
processes_manager.add_process('Photoelectric', 'gamma', 'all')
processes_manager.add_process('Rayleigh', 'gamma', 'all')

range_cuts_manager = GGEMSRangeCutsManager()
range_cuts_manager.set_cut('gamma', 0.1, 'mm', 'all')
range_cuts_manager.set_cut('e-', 0.1, 'mm', 'all')

# 6. SOURCE: Load TIA Map as the activity distribution
voxel_source = GGEMSVoxelizedSource('voxel_source')
voxel_source.set_phantom_source(tia_file)
number_of_particles = int(1e9)
voxel_source.set_number_of_particles(number_of_particles)
voxel_source.set_position(0.0, 0.0, 0.0, 'mm')
# Gamma emissions
voxel_source.set_source_particle_type('gamma')
voxel_source.set_energy_peak(208.4, 'keV', 0.1036)
voxel_source.set_energy_peak(112.9, 'keV', 0.0617)
# Beta- (electron) emissions
voxel_source.set_source_particle_type('e-')
voxel_source.set_energy_peak(149.2, 'keV', 0.793)
voxel_source.set_energy_peak(47.4, 'keV', 0.115)
voxel_source.set_energy_peak(111.1, 'keV', 0.091)

# 7. RUN SIMULATION
ggems = GGEMS()
ggems.opencl_verbose(True)
ggems.material_database_verbose(False)
ggems.navigator_verbose(True)
ggems.source_verbose(True)
ggems.memory_verbose(True)
ggems.process_verbose(True)
ggems.range_cuts_verbose(True)
ggems.random_verbose(True)
ggems.profiling_verbose(True)
ggems.tracking_verbose(False, 0)
seed = 777
ggems.initialize(seed)
ggems.run()

# ------------------------------------------------------------------------------
# 8: Exit safely
dosimetry.delete()
ggems.delete()
exit()
