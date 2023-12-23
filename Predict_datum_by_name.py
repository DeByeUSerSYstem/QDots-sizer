import os
import time
import argparse
from glob import glob
from utils import tools
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# silence tensorflow I, E and W warnings
# or any of {'0', '1', '2', '3'} – 3 silences errors too
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model


def parse_args():
    """
    Parses input file(s) from command line
    """
    parser = argparse.ArgumentParser(prog='Predict_datum_by_name.py',
                    description='''Process experimental data with my CNN.\n
                    Takes as arguments: full path to the diffractogram, the
                    collection wavelength, and (if relevant) the number of
                    rows to be skipped in the diffractogram file''')
    parser.add_argument('diffr',
                        help='Full or Relative path of the diffractogram')
    parser.add_argument('-l', type=float, default=1.5406,
                        help='Collection wavelength, in Å. If omitted, defaults to Kα1-Cu = 1.5406 Å')
    parser.add_argument('--header_rows', type=int, default=0,
                        help='Number of header rows before data starts. If omitted, all rows will be read')

    arg = parser.parse_args()
    return arg


# --------------------------------------------
#  PGM START
# --------------------------------------------
a = time.time()  # time check

# --------------------------------------------
#  PARAMENTERS, FOLDERS AND MODEL DEFINITION
# --------------------------------------------

# parse command line inputs
args = parse_args()

# Set parameters
qmin = 0.99
qMAX = 15.00
labels = np.arange(2.0, 10.1, 0.5)

# set folders paths
pattern_path = args.diffr
pattern_dir = os.path.dirname(args.diffr)
pattern_name = os.path.basename(args.diffr)
name_no_ext = os.path.splitext(pattern_name)[0]

# load model
modelpath = glob('*.h5')[0]
model = load_model(modelpath, compile=True)


# --------------------------------------------
#  LOAD, PREPROCESS AND PREDICT
# --------------------------------------------
# pattern = 'PbSOA_TOL_8nm_0p8Qz_E22_T17keV_SAMPLE_CORR.xye'
print(f'\nOriginal filename: {pattern_name}')
print(f'Using model {modelpath}\n')

# load test data x(2θ) and y(I)
x_exp, y_exp = np.loadtxt(pattern_path, skiprows=args.header_rows,
                          usecols=(0, 1), unpack=True)

# convert 2θ to Q
q_exp = tools.tt2q(x_exp, args.l)

# --- select desired Q values
# if experimental pattern Q-range is same or wider than the std settings:
if np.amin(q_exp) <= qmin and np.amax(q_exp) >= qMAX:
    idx = np.where((q_exp >= qmin) & (q_exp <= qMAX))

    # spline interpolation of test data in the just selected Q-range
    x = np.arange(qmin, qMAX, 0.0028)
    y = np.interp(x, q_exp[idx], y_exp[idx])

    # normalize plot to an area under the curve (auc) = 1000
    auc = np.trapz(y, x=x)
    y_final = y * (1000/auc)

# if the pattern Q-range is narrower than the std settings:
else:
    qmin_exp = q_exp[0] if q_exp[0] >= qmin else qmin
    qMAX_exp = q_exp[-1] if q_exp[-1] <= qMAX else qMAX
    idx = np.where((q_exp >= qmin_exp) & (q_exp <= qMAX_exp))

    # spline interpolation of test data in the just selected Q-range
    x_short = np.arange(qmin_exp, qMAX_exp, 0.0028)
    y_short = np.interp(x_short, q_exp[idx], y_exp[idx])

    # get mean of the first and last 5 values of the diffractogram to use as
    # filler where the pattern is shorter than the std Qrange (0.99-15.00)
    pre_pad_val = np.mean(y_short[:5])
    post_pad_val = np.mean(y_short[-5:])
    tmp_y = np.concatenate((pre_pad_val, y_short, post_pad_val), axis=None)

    # spline interpolation between std-long q_axis (x) and "padded" y
    tmp_q = np.concatenate((qmin_exp, x_short, qMAX_exp), axis=None)
    x = np.arange(qmin, qMAX, 0.0028)
    y = np.interp(x, tmp_q, tmp_y)

    # normalize plot to an area under the curve (auc) = 1000
    auc = np.trapz(y, x=x)
    y_final = y * (1000/auc)

assert len(y_final) == 5004


# save normalised plot as .xy
norm_pattern = np.hstack((x.reshape(-1, 1), y_final.reshape(-1, 1)))
norm_pattern_path = os.path.join(pattern_dir, f'{name_no_ext}_norm.xy')
np.savetxt(norm_pattern_path, norm_pattern, fmt='%.15f')

# generate prediction of test data
prediction = model.predict(y_final.reshape(1, -1))

# get class number and link to relative label
class_n = np.argmax(prediction)  # add axis=1, if multiple test data
class_lbl = labels[class_n]

b = time.time()  # time check


# --------------------------------------------
#  PLOT AND PRINT RESULTS
# --------------------------------------------

# print result
print(f'Predicted size: {class_lbl} nm\n')

# plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title(f'Predicted size: {class_lbl} nm')
ax.plot(x, y_final, color='xkcd:teal', label=pattern_name)
ax.set_xlabel(r'$Q\ (Å^{-1})$')
ax.set_ylabel('Intensitiy (a.u.)')
ax.set_yticks([])
ax.legend(loc='upper right')
fig.tight_layout()

# input user on saving plot
affermative_answer = ['y', 'yes', 'yeah', 'yup']
fname_frendly_lbl = 'p'.join(str(class_lbl).split('.'))

choice = input('Would you like to save the plot? [Y/n]:  ')
if choice.lower() in affermative_answer or choice == '':
    print(f'Plot will be saved in the same folder as the original file, under the name "{name_no_ext}_PredictedSize{fname_frendly_lbl}"')
    folder_choice = input('Would you like to change folder? [y/N]:  ')
    if folder_choice.lower() in affermative_answer:
        user_folder = input('Please, specify new Relative Path here (folder must already exist):  ')
        plt.savefig(os.path.join(user_folder,
                    f'{name_no_ext}_PredictedSize{fname_frendly_lbl}.png'), dpi=200)
    else:
        plt.savefig(os.path.join(pattern_dir,
                    f'{name_no_ext}_PredictedSize{fname_frendly_lbl}.png'), dpi=200)
    plt.show()

# else, just show it
else:
    print('Just showing plot')
    plt.show()


# feedback
print(f'done in {round(b-a)}sec ✓\n')
