import os
import time
import subprocess
import shlex
import shutil
import numpy as np
from random import choice as random_choice
from subprocess import getoutput
from utils import tools
from utils import paths
import argparse


def get_name(strain, std_dev_percentage, avg):
    """
    Extrapolates filenames from building parameters.
    :param strain: multiplicative factor to bulk cell parameter – variable length float
    :param std_dev_percentage: quite self explicative – int
    :param avg: mean average size of the nano-crystals – float
    :return: filename, as a formatted string made of a sequence of "ParameterVALUE_" snippets (i.e. PbS_Str0.9950_SD15_Avg4.0)
    """
    name = f'PbS_Str{strain:.4f}_SD{std_dev_percentage}_Avg{avg:.2f}'
    return name


def create_par_dwa(strain, std_dev_percentage, avg, where_to_save):
    """
    Creates .par and .dwa files in the current folder, by replacing designated snippets with input parameters
    :param strain: multiplicative factor to bulk cell parameter – variable length float
    :param std_dev_percentage: percentage of standard deviation to impose during the simulation creation – int
    :param avg: mean average size of the nano-crystals – float
    :return: nothing. Creates .par and .dwa files in the working directory
    """
    # open .dwa template and save the whole content in "dwa_content" variable
    with open(os.path.join(paths.templates_dir, 'template.dwa'), 'rt') as f:
        dwa_content = f.read()

    # replace the designated strings
    new_name = get_name(strain, std_dev_percentage, avg)
    new_file_path = os.path.join(where_to_save, new_name)
    dwa_content = dwa_content.replace('PLOT1D_NAME_SEED', new_name + '_')
    dwa_content = dwa_content.replace('PAR_FILE_NAME', new_file_path + '.par')
    dwa_content = dwa_content.replace('OUTPUT_NAME_SEED', new_file_path)

    # write the updated dwa_content in a new file.dwa
    # "x" options assures that another file with the same name is not yet
    # present in the folder, if so the system rises an error.
    with open(os.path.join(where_to_save, f'{new_name}.dwa'), 'xt') as f:
        f.write(dwa_content)

    # ============
    # follows the same procedure as above, but for the .par file
    # open .par template and save the whole content in "par_content" variable
    with open(os.path.join(paths.templates_dir, 'template.par'), 'rt') as f:
        par_content = f.read()

    # format parameters according to .par file: pad with zeros after the comma up to 18 characters.
    formatted_strain = str(strain) + '0'*(18 - len(str(strain)))
    formatted_avg = str(avg) + '0'*(18 - len(str(avg)))
    std_dev = avg * std_dev_percentage / 100
    if std_dev == 0:
        formatted_std_dev = '0.00001' + '0'*(18 - 7)  # minimum std_dev allowed from Debussy input is 1E-7
    else:
        formatted_std_dev = str(std_dev) + '0'*(18 - len(str(std_dev)))

    par_content = par_content.replace('_CELLSTRAINCELLST_', formatted_strain)
    par_content = par_content.replace('_STANDARDDEVIATIO_', formatted_std_dev)
    par_content = par_content.replace('_AVERAGEAVERAGEAV_', formatted_avg)

    with open(os.path.join(where_to_save, f'{new_name}.par'), 'xt') as f:
        f.write(par_content)


def get_solvents_tag(list_of_solvents):
    list_with_tags = []
    for name in list_of_solvents:
        list_with_tags.append( name[:3].upper() )
    return list_with_tags


# filename example PbS_Str1.0017_SD20_Avg2.00_Solv1_Conc25.cal
def get_parameter_from_fname(file_name: str, tag: str) -> float:
    """
    Extracts one parameter from the filename: the number following the "Tag_" token
    :param file_name: string divided in tokens by "_" character
    :param tag: string associated to the parameter we want
    :return: numerical value of the wanted parameter
    """
    tag_length = len(tag)
    tokens = os.path.splitext(file_name)[0].split('_')
    for i in tokens:
        if i.lower().startswith(tag.lower()):
            param = i[tag_length:]
            return param
    raise Exception("!!! No label found. Please re-define relative tag. !!!")


def parse_args():
    """
    Parses input variable(s) from command line. Her sets
    """
    prs = argparse.ArgumentParser(prog='1_generate-database.py', 
            description='Creates diffractograms, performs augmentations and stores the whole database in a .npy matrix')
    prs.add_argument('step', type=str, choices=['0.5', '0.25'], help='Choosen step size with which to create the database (in nm).')
    prs.add_argument('qrange', type=str, choices=['Q10', 'Q14'], help='Choosen range in Q to be simulated')
    arg = prs.parse_args()
    return arg

# --------------------------------------------
#  PGM START
# --------------------------------------------
# Parse command line inputs
args = parse_args()
step_size = args.step
q_range = args.qrange

header = f'#  Creating database with: Step-size={step_size}nm, Q-range={q_range}  #'
print('#' * len(header) )
print(header)
print('#' * len(header) )
print(time.asctime())
start = time.time()  # time check


# --------------------------------------------
#  PATHS AND FOLDERS MANAGEMENT
# --------------------------------------------
# set external programs paths
debussy = paths.debussy_PATH
noisymonocol = paths.noisy_PATH

# define reference folder
if step_size == '0.5':
    reference_folder = os.path.join(paths.data_dir, 'Step_05', 'SD5pc_ormore', 'IntermConc_100-65-37-15')  # ! UsualConc_100-70-50-25  -  IntermConc_100-65-37-15
elif step_size == '0.25':
    reference_folder = os.path.join(paths.data_dir, 'Step_025', 'SD5pc_ormore', 'IntermConc_100-65-37-15')  # !  LowConc_100-50-25-5

# create and define subfolders
cal_files_dir = os.path.join(reference_folder, 'cal')
all_other_files_dir = os.path.join(reference_folder, 'ancillary')
diluted_files_dir = os.path.join(reference_folder, 'dilutions', q_range)
noised_files_dir = os.path.join(reference_folder, 'final_noised', q_range)
results_dir = os.path.join(reference_folder, 'results', q_range)

subfolders = [cal_files_dir, all_other_files_dir, diluted_files_dir, noised_files_dir, results_dir]
for subfolder in subfolders:
    tools.initialise_folder(subfolder)


# --------------------------------------------
#  STRUCTURAL PARAMETERS DEFINITION
# --------------------------------------------

# create 3 variables:
# strain = from +0.51% to -0.25% of the bulk cell parameter, in 9 steps --> expressed as multiplicative factor for cell parameter
# standard deviation = 2-5-10-15-20% of the relative average size --> % calculation implemented in the
# -------  "create_dwa_par" function to be directly written into the .par file
# average size = from 2nm to 10nm with 0.5nm step
std_dev = [5, 10, 15, 20]
if step_size == '0.5':
    average_size = np.arange(2.0, 10.1, 0.5).tolist()
    strain = np.linspace(0.9975, 1.0051, 19).tolist()
elif step_size == '0.25':
    average_size = np.arange(2.0, 10.1, 0.25).tolist()
    strain = np.linspace(0.9975, 1.0051, 10).tolist()


# --------------------------------------------
#  CREATE DIFFRACTOGRAMS
# --------------------------------------------

a = time.time()  # time check
# create all the .par and .dwa files.
print('\n=== Creating .dwa and .par files ===')
for s in strain:
    for sd in std_dev:
        for avg in average_size:
            create_par_dwa(s, sd, avg, reference_folder)

# delete std-output file, if already exists.
stdout_log_path = os.path.join(reference_folder, 'debussy-stdout.txt')
if os.path.exists(stdout_log_path):
    os.remove(stdout_log_path)

# get all the names made from the params combinations, add '.dwa' suffix to prepare filename to feed Debussy,
# run all Debussy simulations, and redirect the outputs to 'debussy-stdout.txt' file.
print('=== Running Debussy ===')

z = 0
for s in strain:
    for sd in std_dev:
        for avg in average_size:
            name_seed = get_name(s, sd, avg)
            file_name = name_seed + '.dwa'
            cmd = debussy + f'Debussy {reference_folder}/{file_name}'
            # runs command and returns std-out
            output = getoutput(cmd)
            with open(os.path.join(reference_folder, stdout_log_path), 'a') as o:
                o.write(output + 5*'\n')
            z += 1
            # feedback during creation
            if z % 350 == 0:
                print(f'-- {z} pattern created.')

# Feedback
print('All file created by Debussy')
b = time.time()  # time check
print('Diffractograms created in {:.2f} min'.format( (b-a)/60 ))

# Rearrange just created files in the appropriate folders.
print('=== Arranging files in folders ===')
# removes useless files + arranges files in the designated directories.
for document in os.listdir(reference_folder):
    doc = os.path.join(reference_folder, document)
    if document.endswith('.err') or document.startswith('stage'):
        os.remove(doc)
    elif document.startswith('P'):
        root, extension = os.path.splitext(document)
        if extension in ['.mtx', '.dis', '.sum', '.dwa', '.par']:
            shutil.move(doc, all_other_files_dir)
        elif extension == '.cal':
            dst = root[:-7] + extension
            dst_path = os.path.join(reference_folder, dst)
            os.rename(doc, dst_path)
            shutil.move(dst_path, cal_files_dir)

# removes useless files in project dir, if any.
for rubbish in os.listdir('.'):
    if rubbish.endswith('.err') or rubbish.startswith('stage'):
        os.remove(rubbish)

# Feedback
print(time.asctime())


# --------------------------------------------
#  AUGMENTATION PARAMETERS DEFINITION
# --------------------------------------------

dilutions = [100, 65, 37, 15]  # ! 100, 50, 25, 5  -  100, 70, 50, 25
noise_levels = [200, 30, 8, 3]
simu_wavelength = 0.56  # in Å
solv_wavelength = 0.564162  # in Å
q_sampling = 0.0028  # in Q (0.01430 2θ)
q_min = 0.99
if q_range == 'Q10':
    q_max = 10.27
elif q_range == 'Q14':
    q_max = 15.00


# --------------------------------------------
#  CREATE DILUTIONS AND NORMALISE
# --------------------------------------------
print('\n=== Augmentation 1: add solvents ===')

# ======= Simulations x-axis management =======
# Load a random simulation file  # 23165 points
f = random_choice(os.listdir(cal_files_dir))
simu_x = np.loadtxt(os.path.join(cal_files_dir, f), usecols=0)

# transform to Q
simu_xq = tools.tt2q(simu_x, simu_wavelength)

# find indices of all the points between given Qmin and Qmax
simu_idx = np.where((simu_xq >= q_min) & (simu_xq <= q_max))

# set new x-axis in Q
new_x = np.arange(q_min, q_max, q_sampling)  # control over step size
new_length = new_x.shape[0]
print('Step size: {} q ({:.6f} 20) --> {} points.'.format(q_sampling, tools.q2tt(q_sampling, simu_wavelength), new_length ))

# save it (as it's always the same)
x_final_file_path = os.path.join(reference_folder, f'x-norm_{q_range}')
np.save(x_final_file_path, new_x)

# ======= Solvents management =======

# --- x-axis ---
# load a random .xye file  # 33530 points
s = random_choice(os.listdir(paths.solvents_dir))
solv_x = np.loadtxt(os.path.join(paths.solvents_dir, s), usecols=0)

# transform to Q
solv_xq = tools.tt2q(solv_x, solv_wavelength)

# find indices: 0.99 < Q < 15.00
solv_idx = np.where((solv_xq >= q_min) & (solv_xq <= q_max))

# --- y-axis ---
solvents_order = []
solvents_Y = []
solvents_auc = []

for solvent in os.listdir(paths.solvents_dir):
    solv_y = np.loadtxt(os.path.join(paths.solvents_dir, solvent), usecols=1)

    # spline interpolation of the data on the above defined new x-axis
    Y_solvent = np.interp(new_x, solv_xq[solv_idx], solv_y[solv_idx])

    # calculate the area under the curve (between the plot and the x-axis)
    auc_solvent = np.trapz(x=new_x, y=Y_solvent)

    # store useful data
    solvents_Y.append(Y_solvent)
    solvents_auc.append(auc_solvent)
    solvents_order.append(solvent)
solvents_tag = get_solvents_tag(solvents_order)

# ======= Simulations y-axis management =======
#  (a.k.a. add solvent & normalise @ AUC=1000)
count = 0

# Intensities of all the .cal files
for file in os.listdir(cal_files_dir):
    basename = os.path.splitext(file)[0]

    # fetch third column of .cal file --> y, intensities I.
    simu_y = np.loadtxt(os.path.join(cal_files_dir, file), usecols=2)

    # spline interpolation of the data on the above defined new X-axis
    Y_simu = np.interp(new_x, simu_xq[simu_idx], simu_y[simu_idx])

    # calculate the area under the curve (between the plot and the x-axis)
    auc_nc = np.trapz(x=new_x, y=Y_simu)

    # add each solvent ...
    for solv_name, Y_solv, auc_solv in zip(solvents_tag, solvents_Y, solvents_auc):
        # ... at various ratios
        for r in dilutions:
            ratio_nc = r
            ratio_solv = 100 - ratio_nc

            # total  =    "normalised NC plot * NC_ratio"      +    "normalised solv plot * solv_ratio"
            normed_sum = Y_simu * (1000/auc_nc) * (ratio_nc/100) + Y_solv * (1000/auc_solv) * (ratio_solv/100)

            # save the final normalised pattern
            normed_sums_file_path = os.path.join(diluted_files_dir, f'{basename}_Solv{solv_name}_Conc{r}.cal')
            np.savetxt(normed_sums_file_path, normed_sum, fmt='%.15f')

            count += 1

    # feedback during creation
    if count % 1000 == 0:
        print(f'-- {count} pattern created.')

# Feedback
print(time.asctime())
c = time.time()  # time check
print('Dilutions performed in {:.2f}min'.format( (c-b)/60 ))


# --------------------------------------------
#  INTRODUCE NOISE INTO THE PATTERNS
# --------------------------------------------
print('\n=== Augmentation 2: add noise ===')

for lvl in noise_levels:
    k = 0
    for pattern in os.listdir(diluted_files_dir):
        
        # run noisy
        pattern_path = os.path.join(diluted_files_dir, pattern)
        proc = subprocess.run(shlex.split(f'{noisymonocol} --silent --scale {lvl} {pattern_path}'),
                           text=True, check=True, stdout=subprocess.PIPE)

        if 'WARNING' in proc.stdout:
            print('\n\n!! AN ERROR OCCURRED !!')
            print(proc.stdout)
            print('!! DATA AUGMENTATION PROCESS STOPPED !!')
            sys.exit(1)

        # feedback during creation
        k += 1
        if k % 2000 == 0:
            print(f'-- {k} pattern noised.')

    # move created noisy to dedicated folder
    for file in os.listdir(diluted_files_dir):
        if f'_noisy{lvl}' in file:
            file_path = os.path.join(diluted_files_dir, file)
            shutil.move(file_path, noised_files_dir)
    print('One level of noise applied to the whole dataset.')
    print(time.asctime())  # time check

# Feedback
d = time.time()  # time check
print('Data normalised in {:.2f}min ✓'.format( (d-c)/60 ))


# --------------------------------------------
#  SAVE PATTERNS IN DATA & LABELS MATRICES 
# --------------------------------------------
print('\n=== Creating patterns and labels data-matrices ===')

n = 0
all_names = []
all_files = []
labels_avg = []
for fname in os.listdir(noised_files_dir):

    # get file and relative info
    avg = get_parameter_from_fname(fname, 'avg')
    labels_avg.append(float(avg))
    all_names.append(fname)

    tmp = np.loadtxt(os.path.join(noised_files_dir, fname), dtype='float32').tolist()
    all_files.append(tmp)

    # Feedback
    n += 1
    if n == 500 or n % 5000 == 0:
        print(f'-- {n} files done')

print(f'Matrix created: MIX database consists of {n} patterns.')

# Dimension checks
assert len(all_files) == len(labels_avg) == len(all_names)
assert len(all_files) == len(os.listdir(noised_files_dir))

# Saving matrices as compressed numpy .npy files
print('=== Saving matrix ===')
data = np.asarray(all_files)
np.save(os.path.join(results_dir, f'data_matrix_all_{q_range}_MIX'), data)
np.save(os.path.join(results_dir, f'labels_avg_all_{q_range}_MIX'), labels_avg)
np.save(os.path.join(results_dir, f'fnames_all_{q_range}_MIX'), all_names)


# ### ROUTINE FOR SEPARATE SOLVENTS
# solvents_order = []
# for solvent in os.listdir(paths.solvents_dir):
#     solvents_order.append(solvent)
# solvents_tag = get_solvents_tag(solvents_order)

# select files by solvent
# for tag in solvents_tag:

#     n = 0
#     all_names = []
#     all_files = []
#     labels_avg = []
#     # labels_sd = []
#     for fname in os.listdir(noised_files_dir):
        
#         # get desired info from filename handling
#         solv = get_parameter_from_fname(fname, 'solv')
#         if solv == tag:

#             # get file and relative info
#             avg = get_parameter_from_fname(fname, 'avg')
#             labels_avg.append(float(avg))
#             all_names.append(fname)
   
#             tmp = np.loadtxt(os.path.join(noised_files_dir, fname), dtype='float32').tolist()
#             all_files.append(tmp)
            
#             # Feedback
#             n += 1
#             if n == 500 or n % 1000 == 0:
#                 print(f'-- {n} files done')
    
#     print(f'Matrix created: {tag} database consists of {n} patterns.')

#     # Dimension checks
#     assert len(all_files) == len(labels_avg) == len(all_names)
#     assert len(all_files) == len(os.listdir(noised_files_dir))/2
#     assert len(labels_avg) == len(os.listdir(noised_files_dir))/2  # == len(labels_sd)

#     # Saving matrices as compressed numpy .npy files
#     print('=== Saving matrix ===')
#     data = np.asarray(all_files)
#     np.save(os.path.join(results_dir, f'data_matrix_all_{q_range}_{tag}'), data)
#     np.save(os.path.join(results_dir, f'labels_avg_all_{q_range}_{tag}'), labels_avg)
#     np.save(os.path.join(results_dir, f'fnames_all_{q_range}_{tag}'), all_names)


# Feedback
end = time.time()  # time check
print('\n\n==> Dataset created, augmented, normalised and saved! <==')
print('All done in {:.0f}min  ✓'.format( (end-start)/60 ))
print(time.asctime())  # time check
print('\n\n')

