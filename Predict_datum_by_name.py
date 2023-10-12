import os
# silence tensorflow I, E and W warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any of {'0', '1', '3'} – 3 silences errors too
import time
from keras.models import load_model
import numpy as np
import argparse
from utils import tools
from utils import paths
import sys
import matplotlib.pyplot as plt


def parse_args():
    """
    Parses input file(s) from command line
    """
    parser = argparse.ArgumentParser(prog='5_process-real-datum_by-name.py',
                                     description='Process real data with my CNN')
    parser.add_argument('diffr', help='Relative Path to Diffractogram of interest')
    parser.add_argument('-l', type=float, default=[0.56], help='Syncrhotron collection wavelength, in Å. DEFAULT=0.56')
    parser.add_argument('--header_rows', type=int, default=0, help='Header rows before data starts')
    arg = parser.parse_args()
    return arg

# --------------------------------------------
#  PGM START
# --------------------------------------------
a = time.time()  # time check

# --------------------------------------------
#  PARAMENTERS, FOLDERS AND MODEL DEFINITION
# --------------------------------------------

# Parse command line inputs
args = parse_args()

# Set parameters
qmin = 0.99
qMAX = 15.00
labels = np.arange(2.0, 10.1, 0.5)

# set folders paths
pattern_path = args.diffr
pattern_name = os.path.basename(args.diffr)
test_data_dir = os.path.dirname(args.diffr)
model_dir = os.path.join(paths.project_dir, 'good-models')

# load model
modelpath = os.path.join(model_dir, 'mix', 'Step05_Q14_R_MIX_1.h5')
model = load_model(modelpath , compile=True)


# --------------------------------------------
#  LOAD, PREPROCESS AND PREDICT
# --------------------------------------------
#pattern = 'PbSOA_TOL_8nm_0p8Qz_E22_T17keV_SAMPLE_CORR.xye'
print(f'\nOriginal filename: {pattern_name}')

# load test data x(2θ) and y(I)
x_real, y_real = np.loadtxt(pattern_path, skiprows=args.header_rows, usecols=(0, 1), unpack=True)

# convert 2θ to Q
q_real = tools.tt2q(x_real, args.l)

# select desired Q values
if np.amin(q_real) <= qmin and np.amax(q_real) >= qMAX:
    idx = np.where((q_real > qmin) & (q_real < qMAX))
else:
    print('Data Q-range shorter than allowed (14)')
    sys.exit(1)

# spline interpolation of test data in the just selected Q-range
x = np.arange(qmin, qMAX, 0.0028)
y = np.interp(x, q_real[idx], y_real[idx])

# normalize plot to and area under the curve (auc) = 1000
auc = np.trapz(y, x=x)
y_norm = y * (1000/auc)

# save normalised plot as .xy
#norm_pattern = np.hstack((x.reshape(-1, 1), y_norm.reshape(-1, 1)))
#np.savetxt(os.path.join(test_data_dir, 'pattern_normalised'), norm_pattern, fmt='%.15f')

# generate prediction of test data
prediction = model.predict(y_norm.reshape(1, -1))

# get class number and link to relative label
class_n = np.argmax(prediction)  # axis=1, if multiple test data
class_lbl = labels[class_n]


# --------------------------------------------
#  PLOT AND PRINT RESULTS
# --------------------------------------------

# print result
print(f'Predicted size: {class_lbl} nm')

# plot
fig, ax = plt.subplots(figsize=(15, 7))
ax.set_title(f'Predicted size: {class_lbl} nm')
ax.plot(x, y_norm, color='xkcd:teal', label=pattern_name)
ax.set_xlabel(r'$Q\ (Å^{-1})$')
ax.set_ylabel('Intensitiy (a.u.)')
ax.set_yticks([])
ax.legend(loc='upper right')
fig.tight_layout()

# Input user on saving plot
affermative = ['y', 'yes', 'yeah', 'yup']
choice = input('Would you like to save the plot? [Y/n]:  ')
if choice.lower() in affermative or choice is None:
	print(f'Plot will be saved in the same folder as the original under the name "{pattern_name}_PredAvgSize{class_lbl}"')
	folder_choice = input('Would you like to change folder? [Y/n]:  ')
	if folder_choice.lower() in affermative:
		user_folder = input('Please, specify Relative Path here:  ')
		plt.savefig(os.path.join(user_folder, f'{pattern_name[:-4]}_PredictedSize{class_lbl}.png'), dpi=200)
	else:
		plt.savefig(os.path.join(test_data_dir, f'{pattern_name[:-4]}_PredictedSize{class_lbl}.png'), dpi=200)
# else, just show it
else:
    print('Just showing plot')
    plt.show()


# feedback
b = time.time()  # time check
print(f'done in {round(b-a)}sec ✓')

