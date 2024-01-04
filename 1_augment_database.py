#!/usr/bin/python3

import os
import sys
import time
import shlex
import shutil
import argparse
import subprocess
import numpy as np
from utils import paths
from utils import tools
from random import choice as random_choice


def get_solvents_tag(list_of_solvents):
    list_with_tags = []
    for name in list_of_solvents:
        list_with_tags.append(name[:3].upper())
    return list_with_tags


# filename example PbS_Str1.0017_SD20_Avg2.00_Solv1_Conc25.cal
def get_parameter_from_fname(file_name: str, tag: str) -> str:
    """
    From the filename, extracts the numerical parameter following "Tag_" token.
    :param file_name: string divided in tokens by "_" character
    :param tag: string denoting the parameter we want
    :return: numerical value of the wanted parameter
    """
    tag_length = len(tag)
    tokens = os.path.splitext(file_name)[0].split('_')
    for i in tokens:
        if i.lower().startswith(tag.lower()):
            return i[tag_length:]
    raise Exception("!!! No label found. Please re-define relative tag. !!!")


def parse_args():
    """
    Parses input file(s) from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation_folder',
                        help='Full path of the folder containing the diffractograms you wish to augment.')

    arg = parser.parse_args()
    return arg


# --------------------------------------------
#  PGM START
# --------------------------------------------

# """
# *** DISCLAIMER ***

# This script assumes that there already exists a folder where the user has
# a collection of .cal files, namely diffractograms previously simulated with
# Debussy and selected to be the basis of the database and thus be subjected
# to augmentation.

# The augmentation procedure follows:
# """

header = f'#  Augmenting database  #'
print('#' * len(header))
print(header)
print('#' * len(header))
print(time.asctime())
start = time.time()  # time check


# --------------------------------------------
#  PATHS AND FOLDERS MANAGEMENT
# --------------------------------------------
# set external programs paths
noisymonocol = paths.noisy_PATH

# folder containing all the .cal files = simulated patterns generated with Debussy
args = parse_args()
cal_files_dir = args.simulation_folder

# create and define subfolders
diluted_files_dir = os.path.join(paths.data_dir, 'dilutions')
noised_files_dir = os.path.join(paths.data_dir, 'final_noised')
results_dir = os.path.join(paths.data_dir, 'results')

subfolders = [diluted_files_dir, noised_files_dir, results_dir]
for subfolder in subfolders:
    tools.initialise_folder(subfolder)


# --------------------------------------------
#  AUGMENTATION PARAMETERS DEFINITION
# --------------------------------------------
a = time.time()  # time check

dilutions = [100, 65, 37, 15]  # % of NCs in a NC-Solvent linear combination
noise_levels = [200, 30, 8, 3]  # scale parameter of the "noisy" program
simu_wavelength = 0.56  # in Å
solv_wavelength = 0.564162  # in Å
q_step = 0.0028  # in Q (0.01430 2θ)
q_min = 0.99
q_max = 15.00


# --------------------------------------------
#  CREATE DILUTIONS AND NORMALISE
# --------------------------------------------
print('\n=== Augmentation 1: add solvent(s) ===')

# ======= Simulations x-axis management =======
# Load a random simulation file  # 23165 points
f = random_choice(os.listdir(cal_files_dir))
simu_x = np.loadtxt(os.path.join(cal_files_dir, f), usecols=0)

# transform 2Θ in Q
simu_xq = tools.tt2q(simu_x, simu_wavelength)

# find indices of all the points between given Qmin and Qmax
simu_idx = np.where((simu_xq >= q_min) & (simu_xq <= q_max))

# set new x-axis in Q
new_x = np.arange(q_min, q_max, q_step)  # control over step size
# new_x = np.linspace(q_min, q_max, 5004)  # control over n' of point
new_length = new_x.shape[0]
print('Step size: {} Q ({:.4f} 20) --> {} points.'
      .format(q_step, tools.q2tt(q_step, simu_wavelength), new_length))

# save it for reference(it will always be the same)
x_final_file_path = os.path.join(paths.data_dir, f'x-norm_Q15')
np.save(x_final_file_path, new_x)


# ======= Solvent(s) management =======

# --- x-axis ---
# load a random .xye file  # 33530 points
s = random_choice(os.listdir(paths.solvents_dir))
solv_x = np.loadtxt(os.path.join(paths.solvents_dir, s), usecols=0)

# transform 2Θ in Q
solv_xq = tools.tt2q(solv_x, solv_wavelength)

# find indices between given Qmin and Qmax: 0.99 ≤ Q ≤ 15.00
solv_idx = np.where((solv_xq >= q_min) & (solv_xq <= q_max))

# --- y-axis ---
solvents_Y = []
solvents_auc = []
solvents_order = []

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

# Intensities of all the simulation(.cal) files
for file in os.listdir(cal_files_dir):
    basename = os.path.splitext(file)[0]

    # fetch third column of .cal file --> y, intensities I
    simu_y = np.loadtxt(os.path.join(cal_files_dir, file), usecols=2)

    # spline interpolation of the data on the above defined new X-axis
    Y_simu = np.interp(new_x, simu_xq[simu_idx], simu_y[simu_idx])

    # calculate the area under the curve (between the plot and the x-axis)
    auc_nc = np.trapz(x=new_x, y=Y_simu)

    # add each solvent at various ratios
    for solv_name, Y_solv, auc_solv in zip(solvents_tag, solvents_Y, solvents_auc):
        for nc_ratio in dilutions:
            solv_ratio = 100 - nc_ratio

            # total    =    "normalised NC plot * NC_ratio"      +    "normalised solv plot * solv_ratio"
            normed_sum = Y_simu * (1000/auc_nc) * (nc_ratio/100) + Y_solv * (1000/auc_solv) * (solv_ratio/100)

            # save the final normalised pattern
            fpath_normed_sum = os.path.join(diluted_files_dir,
                                            f'{basename}_Solv{solv_name}_Conc{nc_ratio}.cal')
            np.savetxt(fpath_normed_sum, normed_sum, fmt='%.15f')

            count += 1

    # feedback during creation
    if count % 3000 == 0:
        print(f'-- {count} pattern created.')

# Feedback
print(time.asctime())
b = time.time()  # time check
print('Dilutions performed in {:.2f}min'.format((b-a)/60))


# --------------------------------------------
#  INTRODUCTION OF NOISE INTO THE PATTERNS
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

    # move created noised pattern to dedicated folder
    for file in os.listdir(diluted_files_dir):
        if f'_noisy{lvl}' in file:
            file_path = os.path.join(diluted_files_dir, file)
            shutil.move(file_path, noised_files_dir)
    print('One level of noise applied to the whole dataset.')
    print(time.asctime())  # time check

# Feedback
c = time.time()  # time check
print('Data normalised in {:.2f}min ✓'.format((c-b)/60))


# -------------------------------------------------
#  SAVE PATTERNS IN DATA & LABELS & NAMES MATRICES
# -------------------------------------------------
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

    tmp = np.loadtxt(os.path.join(noised_files_dir, fname), dtype='float32')
    all_files.append(tmp)

    # Feedback
    n += 1
    if n == 500 or n % 5000 == 0:
        print(f'-- {n} files done')

print(f'Matrix created: MIX database consists of {n} patterns.')

# Dimensions check
assert len(all_files) == len(labels_avg) == len(all_names)
assert len(all_files) == len(os.listdir(noised_files_dir))

# Saving matrices as compressed numpy .npy files
print('=== Saving matrix ===')
data = np.asarray(all_files)
np.save(os.path.join(results_dir, f'data_matrix'), data)
np.save(os.path.join(results_dir, f'labels'), labels_avg)
np.save(os.path.join(results_dir, f'fnames'), all_names)

# Feedback
end = time.time()  # time check
print('\n\n==> Dataset created, augmented, normalised and saved! <==')
print('All done in {:.0f}min  ✓'.format((end-start)/60))
print(time.asctime())  # time check
print('\n\n')
