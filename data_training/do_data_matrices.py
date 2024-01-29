#!/usr/bin/python3

import os
import sys
import numpy as np


# filename example:  PbS_Str1.0017_SD20_Avg2.00_Solv1_Conc25.cal
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


# -------------------------------------------------
#  SAVE PATTERNS IN DATA & LABELS & NAMES MATRICES
# -------------------------------------------------

tar_extraction_dir = sys.argv[1]

all_names = []
all_files = []
labels_avg = []
n = 0

for sub_folder in os.listdir(tar_extraction_dir):
    for fname in os.listdir(os.path.join(tar_extraction_dir, sub_folder)):
        if fname.startswith('p'.upper()):
            # get file and relative info
            avg = get_parameter_from_fname(fname, 'avg')
            labels_avg.append(float(avg))
            all_names.append(fname)

            tmp = np.loadtxt(os.path.join(tar_extraction_dir, sub_folder, fname),
                             dtype='float32')
            all_files.append(tmp)

            # Feedback
            n += 1
            if n == 500 or n % 5000 == 0:
                print(f'-- {n} files done')

# Dimensions check
assert len(all_files) == len(labels_avg) == len(all_names)
assert len(all_files) == n

# Saving matrices as compressed numpy .npy files
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'DATA')
os.makedirs(data_dir)

data = np.asarray(all_files)
np.save(os.path.join(data_dir, f'data_matrix'), data)
np.save(os.path.join(data_dir, f'labels'), labels_avg)
np.save(os.path.join(data_dir, f'fnames'), all_names)
