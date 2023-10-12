#!/usr/bin/python3

import os
# silence tensorflow I, E and W warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any of {'0', '1', '3'} – 3 silences errors too
from glob import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.compat.v1 import reset_default_graph
from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense
from keras.optimizers.optimizer_v2 import adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
from utils import paths
from utils import tools
import argparse


def create_encoders(all_labels):
    """
    Creates and fits encoders.
    :param all_labels: numpy-array or list of all the dataset's labels
    :return: two fitted encoders
    """
    # gather a list of all labels
    s = set(all_labels)
    labels = np.asarray(list(s))

    # create label encoder: transforms our float-labels into integer values
    label_encoder = LabelEncoder()
    labels_as_int = label_encoder.fit_transform(labels)

    # create one-hot encoder from the integers
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    labels_as_int = labels_as_int.reshape(-1, 1)
    one_hot_encoder.fit(labels_as_int)

    return label_encoder, one_hot_encoder


def one_hot_encoding(x, tag, label_encoder=None, one_hot_encoder=None):
    """
    Label encoding/decoding: original categorical values --> one-hot vector, or vice-versa.
    :param x: labels to be transformed (numpy-array or list)
    :param tag: string indicating required transformation. Allowed values: "encode", "decode"
    :param label_encoder: *LEAVE DEFAULT* encoder from categorical to integers, and vice-versa
    :param one_hot_encoder: *LEAVE DEFAULT* encoder for integer to on-hot vector, and vice-versa
    :return: encoded or decoded list of labels
    """
    # set of permitted values for param "tag"
    valid = {'encode', 'decode'}
    if tag not in valid:
        raise ValueError('Tag parameter must be explicit. Accepted values: "encode", "decode".')

    # reference encoders
    if label_encoder is None:
        label_encoder = lbl_enc
    if one_hot_encoder is None:
        one_hot_encoder = onehot_enc

    # encode labels
    if tag == 'encode':
        as_int = label_encoder.transform(x)
        as_int = as_int.reshape(-1, 1)
        encoded = one_hot_encoder.transform(as_int)
        return encoded

    # decode labels
    elif tag == 'decode':
        as_int = [np.argmax(j) for j in x]
        original = label_encoder.inverse_transform(as_int)
        return original


def get_learning_curves(dataframe, round_n):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.suptitle(f'round {round_n}')

    # Loss subplot (ax1)
    ax1.plot(dataframe['loss'], color='tab:blue', marker='.', label='train')
    ax1.plot(dataframe['val_loss'], color='tab:orange', marker='.', label='val')
    # ax1.set_ylim(-0.01, ax1.set_ylim()[1])
    ax1.set_ylim(-0.01, 0.30)
    ax1.grid(color='0.95')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right')

    # Accuracy subplot (ax2)
    ax2.plot(dataframe['accuracy'], color='tab:blue', marker='.',  label='train')
    ax2.plot(dataframe['val_accuracy'], color='tab:orange', marker='.',  label='val')
    # ax2.set_ylim(ax2.set_ylim()[0], 1.01)
    ax2.set_ylim(0.70, 1.01)
    ax2.grid(color='0.95')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_xlabel('epochs', fontsize=12)
    ax2.legend(loc='lower right')

    # Adjust figure spaces
    fig.tight_layout()


# cmaps: -Blues-, GnBu,,, PuBu, YlGn, Greens, YlOrRd (!)
def get_conf_matrix(true_val, pred_val, round_n, threshold):
    
    # set labels manually
    threshold = float( threshold[0] + '.' + threshold[1:] )
    axis_labels = [str(i) for i in np.arange(2.0, 10.5, threshold)]

    y_true = lbl_enc.transform(true_val)
    y_pred = lbl_enc.transform(pred_val)

    # compute the confusion matrix
    conf_mtx = confusion_matrix(y_true, y_pred, normalize='true')

    # save numerical log of the confusion matrix
    conf_mtx_log = os.path.join(reference_folder, 'ConfMatrix', f'Step{step}_{q_range}_{kernel_name}_{solv}_ConfMtx_log_{k}.txt')
    np.savetxt(conf_mtx_log, conf_mtx, fmt='%.5f')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(conf_mtx, cmap='GnBu',
            norm=plt_colors.LogNorm(vmin=0.01, vmax=1))

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Normalized Count', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Set tick labels and title
    ax.set_xticks(np.arange( len(axis_labels) ))
    ax.set_yticks(np.arange( len(axis_labels) ))
    ax.set_xticklabels(axis_labels, fontsize=10)
    ax.set_yticklabels(axis_labels, fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title(f'Confusion Matrix – {kernel_name} Round {round_n}', fontsize=14)


    # Text annotations
    for i in range(len(axis_labels)):
        for j in range(len(axis_labels)):
            value = conf_mtx[i, j]
            text = f'{value:.2f}' if value != 0 else ''
            color = 'white' if value > 0.3 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)                           

    # Set aspect ratio
    ax.set_aspect('equal')


def count_errors(file_name, true_val, pred_val, threshold):
    threshold = float( threshold[0] + '.' + threshold[1:] )
    one_class_wrong = 0
    two_class_wrong = 0
    badly_wrong = 0
    total = len(pred_val)

    mistakes = []
    for i in range(total):
        if true_val[i] != pred_val[i]:
            mistakes.append([file_name[i], true_val[i], pred_val[i]])
            diff = abs(true_val[i] - pred_val[i])
            if diff == threshold:
                one_class_wrong += 1
            elif diff == 2*threshold:
                two_class_wrong += 1
            elif diff > 2*threshold:
                badly_wrong += 1
    
    # save info on all the mistaken patterns
    error_list = pd.DataFrame(mistakes, columns=['file name', 'true size', 'prediction'])
    error_list.to_csv( os.path.join(reference_folder, 'Summary-of-predictions', 
                f'Step{step}_{q_range}_{kernel_name}_{solv}_error_list_{k}.csv') )
    
    return [one_class_wrong, two_class_wrong, badly_wrong, total]


def show_error_report(list_of_error_stats):
    for n, j in enumerate(list_of_error_stats):
        one_class_err = j[0]
        two_class_err = j[1]
        bad_err = j[2]
        tot = j[3]
        one_class_err_percentage = (one_class_err / tot) * 100
        two_class_err_precentage = (two_class_err / tot) * 100
        bad_err_pergentage = (bad_err / tot) * 100
        err_sum = one_class_err + two_class_err + bad_err
        all_err_percentage = (err_sum / tot) * 100
        print()
        print(f'In round {n}:')
        print('Model was wrong {} times over {} ({:0.2f}%)'.format( err_sum, tot, all_err_percentage ))
        if two_class_err != 0 or bad_err != 0:
            print('Of which: {} ({:0.2f}%) errors within 1 class, {} ({:0.2f}%) within two classes, and {} ({:0.2f}%) huge errors'
                .format(one_class_err, one_class_err_percentage, two_class_err, two_class_err_precentage, bad_err, bad_err_pergentage))


# Create and "error per class" histogram
def do_size_error_histograms(kname, rep, accuracies):
    # create dataframe with all classes
    data_size = pd.DataFrame()
    data_size['classes'] = np.arange(2.0, 10.1, 0.5)
    maximum = 0

    # per each k-round:
    for r in range(rep):

        # load error list
        fname = glob( os.path.join(reference_folder, 'Summary-of-predictions', f'*_{kname}_*error_list_{r}.csv') )[0]
        error_list = np.loadtxt(fname, delimiter=',', usecols=2, skiprows=1)
        
        # count occurrncies of every value and store in a df
        values, counts = np.unique(error_list, return_counts=True)
        df = pd.DataFrame()
        df['classes'] = values
        df['occurrences'] = counts

        # normalize counts to 100%
        assert len(error_list) == df['occurrences'].sum()
        df['occurrences'] = df['occurrences'] / df['occurrences'].sum() * 100
        
        # Raname column to store it in the collective DataFrame
        col_name = 'round {} – Acc:{:.3f}%, err:{}/{}'.format( r, accuracies[r]*100, len(error_list), len(z[test]) )
        df = df.rename(columns={'occurrences': col_name})

        # add results to main df
        data_size = data_size.join(df.set_index('classes'), on='classes').fillna(0)
        maximum = df.max(axis=0)[col_name] if df.max(axis=0)[col_name] > maximum else maximum
        del df

    # save error histogram data_size
    data_size.to_csv(os.path.join(reference_folder, 'errorHistogr', f'{os.path.basename(fname)[:-17]}_error_counts_size.csv'))

    # plot all columns against 'classes' index
    title = f'Error evolution per size for model: {kname} {solv}'
    plot = data_size.plot(x='classes', kind='bar', subplots=True, sharex=False, layout=(2, 3), figsize=(17, 8), 
                title=title, fontsize=9, ylim=[0, maximum+(maximum/100*5)], xlabel='classes', ylabel='error %', legend=False)
    plt.tight_layout()


def do_conc_error_histograms(kname, rep, accuracies):
    # Prepare collective DataFrame
    data_conc = pd.DataFrame()
    data_conc['Concentrations'] = ['Conc100', 'Conc65', 'Conc37', 'Conc15']  # !
    maximum = 0

    # list all needed files
    file_names = sorted(glob(os.path.join(reference_folder, 'Summary-of-predictions', f'*_{kname}_*error_list_*.csv')))
    total_files = len(file_names)
    assert total_files == rep

    # Iterate through each file and store the corresponding frequencies
    for i, file_name in enumerate(file_names):
        # Load current CSV file
        data = pd.read_csv(file_name)
        column_data = data['file name']
        
        # Initialize the frequency dictionary for the current file
        frequency = {'Conc100': 0, 'Conc65': 0, 'Conc37': 0, 'Conc15': 0}  # !
        
        # Iterate through each row of the column to check for the serach string
        for row in column_data:
            for search_string in frequency.keys():
                if search_string in row:
                    frequency[search_string] += 1
        
        # Convert the frequency dictionary to a DataFrame for easier plotting
        frequency_df = pd.DataFrame({'String': list(frequency.keys()), 'Frequency': list(frequency.values())})

        # Calculate the percentage relative to the total number of rows in the current file
        total_rows_current_file = len(column_data)
        assert total_rows_current_file == frequency_df['Frequency'].sum()
        frequency_df['Frequency'] = (frequency_df['Frequency'] / total_rows_current_file) * 100

        # Raname column to store it in the collective DataFrame
        col_name = 'round {} – Acc:{:.3f}%, err:{}/{}'.format( i, accuracies[i]*100, total_rows_current_file, len(z[test]) )
        frequency_df = frequency_df.rename(columns={'Frequency': col_name})
        
        # Add it
        data_conc = data_conc.join(frequency_df.set_index('String'), on='Concentrations').fillna(0)
        maximum = frequency_df.max(axis=0)[col_name] if frequency_df.max(axis=0)[col_name] > maximum else maximum
        del frequency_df

    # save error histogram data_conc
    data_conc.to_csv(os.path.join(reference_folder, 'errorHistogr', f'{os.path.basename(file_name)[:-17]}_error_counts_conc.csv'))

    # plot all columns against 'Concentrations' index
    title = f'Error evolution per concentration level for model: {kname} {solv}'
    ax = data_conc.plot(x='Concentrations', kind='bar', subplots=True, 
                    sharex=False, layout=(2, 3), figsize=(12, 6), title=title, 
                    fontsize=9, ylim=[0, maximum+(maximum/100*5)], 
                    xlabel='Concentrations', rot=0, ylabel='error %', legend=False)
    plt.tight_layout()


def set_kernel_size(t):
    t = str(t).upper()
    kernel_dims = {}
    kernel_dims['R'] = [10, 5, 4, 3, 2]  # Q14
    kernel_dims['S'] = [10, 5, 4, 3, 2]
    kernel_dims['T'] = [6, 5, 4, 3, 2]
    kernel_dims['Y'] = [9, 5, 4, 3, 2]  # Q10
    kernel_dims['Z'] = [9, 5, 4, 3, 2]
    kernel_dims['ZA'] = [9, 3, 3, 3, 2]
    return kernel_dims[t]


def parse_args():
    """
    Parses input variable(s) from command line
    """
    prs = argparse.ArgumentParser(prog='4k_tinyCNN.py', description='a-CNN for cNCs PbS size classification')
    prs.add_argument('step', type=str, choices=['0.5', '0.25'], help='Choosen step size with which to create the database (in nm).')
    prs.add_argument('qrange', type=str, choices=['Q10', 'Q14'], help='Choosen range in Q to be simulated')
    prs.add_argument('kernel', type=str, help='given kernel name')
    prs.add_argument('solvent', type=str, choices=['HEX', 'TOL', 'MIX'], help='tag for selected solvent of interest')
    arg = prs.parse_args()
    return arg


# --------------------------------------------
#  PGM START
# --------------------------------------------
print(time.asctime())
start = time.time()  # time check

# Clear Keras and TF session, if run previously
clear_session()
reset_default_graph()

# Parse command line inputs
args = parse_args()
step = ''.join(args.step.split('.'))
q_range = args.qrange.upper()
kernel_name = args.kernel.upper()  # set kernel to be used
solv = args.solvent.upper()  # set TrSet created diluting with which solvent

# set parameter
print(f'TRAINING FOR STEP={step}nm {q_range} WITH KERNEL {kernel_name} IN {solv}.')

# Set reference folder, and folders reqired for diagnostics storage
data_folder = os.path.join(paths.data_dir, f'Step_{step}',
                'SD5pc_ormore', 'IntermConc_100-65-37-15', 'results', q_range)  # ! UsualConc_100-70-50-25  -  IntermConc_100-65-37-15  -  LowConc_100-50-25-5
reference_folder = os.path.join(data_folder, '5_layers')
tools.set_required_subfolders(reference_folder, 'logs', 'models', 'learning-curves', 'ConfMatrix', 'errorHistogr', 'Summary-of-predictions')


# --------------------------------------------
#  LOAD DATA AND ENCODE LABELS
# --------------------------------------------

# Load data and relative desired labels
print('=== Loading data ===')
X_filename = glob(os.path.join(data_folder, f'data_matrix*{q_range}_{solv}.npy'))[0]
y_filename = glob(os.path.join(data_folder, f'labels_avg*{q_range}_{solv}.npy'))[0]
z_filename = glob(os.path.join(data_folder, f'fnames*{q_range}_{solv}.npy'))[0]
X = np.load(os.path.join(data_folder, X_filename))
y = np.load(os.path.join(data_folder, y_filename))
z = np.load(os.path.join(data_folder, z_filename))


# create encoders
lbl_enc, onehot_enc = create_encoders(y)
# encode labels
y_hot = one_hot_encoding(y, 'encode')


# feedback
print(time.asctime())
a = time.time()  # time check
print(f'Data loaded in {round(a-start)}sec')


# --------------------------------------------
#  PERFORM TRAINING AND CROSS-VALIDATION
# --------------------------------------------

# set cross-validation
folds = 5  # number of k-folds
k_fold = KFold(n_splits=folds, shuffle=True, random_state=5)

# create auxilliary arrays
accuracy = []
errors_count = []


# Run cross validation and train a-CNN each time in loop
for k, (train, test) in enumerate(k_fold.split(X, y_hot, z)):
    b = time.time()

    # Define Training and Network parameters
    lr = 0.001
    batch_size = 256
    n_epochs = 400
    patience = 20

    # kernels
    k1, k2, k3, k4, k5 = set_kernel_size(kernel_name)
    if kernel_name == 'S' or kernel_name == 'Z':
        stride1 = 2
    elif kernel_name == 'Za':
        stride1 = 3
    else:
        stride1 = k1

    # Pattern length --> Input size
    if q_range == 'Q10':
        inp_len = 3315
    elif q_range == 'Q14':
        inp_len = 5004

    # Output length --> N° of classes
    if step == '05':
        n_classes = 17
    elif step == '025':
        n_classes = 33


    # Define Keras Model
    # https://doi.org/10.1038/s41524-019-0196-x
    model = Sequential(name='tinyCNN')

    model.add(Conv1D(32, k1, strides=stride1, padding='same', input_shape=(inp_len, 1), activation='relu'))
    model.add(Conv1D(32, k2, strides=k2, padding='same', activation='relu'))
    model.add(Conv1D(32, k3, strides=k3, padding='same', activation='relu'))
    model.add(Conv1D(32, k4, strides=k4, padding='same', activation='relu'))
    model.add(Conv1D(32, k5, strides=k5, padding='same', activation='relu'))    
    model.add(GlobalAveragePooling1D())
    model.add(Dense(n_classes, activation='softmax'))

    # Print model summary (just once)
    if k == 0:
        print("\nModel summary:")
        print(model.summary())
        print('')
        print('kernels check: kernel dims: {} {} {} {} {}, stride1: {}, k_name {}'
                .format(k1, k2, k3, k4, k5, stride1, kernel_name) )
        print('')

    # optimizer
    opt = adam.Adam()

    # choose early stop
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)

    # compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit model
    print(f'\n=== Training, round {k} === {time.asctime()}')
    hist = model.fit(X[train], y_hot[train], batch_size=batch_size, epochs=n_epochs, verbose=0,
             validation_data=(X[test], y_hot[test]), callbacks=[early_stop], shuffle=True)

    # compute loss and accuracy for each k validation
    loss, acc = model.evaluate(X[test], y_hot[test], verbose=0)
    accuracy.append(acc)

    # compute (and decode) model predictions
    encoded_prediction = model.predict(X[test], verbose=0)
    prediction = one_hot_encoding(encoded_prediction, 'decode')

    # feedback
    c = time.time()  # time check
    print('This training round took {:.2f}min'.format( (c-b)/60) )


    # --------------------------------------------
    #  DIAGNOSTICS
    # --------------------------------------------

    # Save training log
    log = pd.DataFrame(hist.history)
    log.to_csv(os.path.join(reference_folder, 'logs', f'log_Step{step}_{q_range}_{kernel_name}_{solv}_{k}.txt'), sep='\t', index=False)

    # Save model
    model.save(os.path.join(reference_folder, 'models', f'Step{step}_{q_range}_{kernel_name}_{solv}_{k}.h5'))

    # Learning Curves: training VS eval Loss & Accuracy
    get_learning_curves(log, k)
    plt.savefig(os.path.join(reference_folder, 'learning-curves', f'lc_Step{step}_{q_range}_{kernel_name}_{solv}_{k}'), dpi=200)
    plt.close()

    # Confusion Matrix: true VS predicted values
    assert len(test) == len(prediction)
    get_conf_matrix(y[test], prediction, k, step)
    # save plot 
    plt.savefig(os.path.join(reference_folder, 'ConfMatrix', f'cm_Step{step}_{q_range}_{kernel_name}_{solv}_{k}'), dpi=700)
    plt.close()

    # Save prdictions, labels and filenmes
    all_predictions = pd.DataFrame()
    all_predictions['File names'] = z[test]
    all_predictions['Labels'] = y[test]
    all_predictions['Predictions'] = prediction
    summary_pred_path = os.path.join(reference_folder, 'Summary-of-predictions', f'Step{step}_{q_range}_{kernel_name}_{solv}_all_predictions_{k}.csv')
    all_predictions.to_csv(summary_pred_path)

    # Gather errors details
    errors_count.append( count_errors(z[test], y[test], prediction, step) )


# --------------------------------------------
#  PRINT RESULTS
# --------------------------------------------

# Accuracy report
accuracy = np.asarray(accuracy)
print('\nTraining with cross-validation ended.')
print(f'Acuracies for each round are: {accuracy*100}')
print('--> Mean Cross-val accuracy {:.3f}% <--'.format(np.mean(accuracy)*100))

# Error-per-size histogram image
do_size_error_histograms(kernel_name, folds, accuracy)
plt.savefig(os.path.join(reference_folder, 'errorHistogr', f'SizeError_per_class_{kernel_name}'), dpi=200)
plt.close()

# Error-per-conc histogram image
do_conc_error_histograms(kernel_name, folds, accuracy)
plt.savefig(os.path.join(reference_folder, 'errorHistogr', f'ConcError_per_class_{kernel_name}'), dpi=200)
plt.close()

# feedback
print('\n', '-'*66, '\n')
print(time.asctime())
end = time.time()  # time check
print('The whole process took {:.0f}min'.format( (end-start)/60) )

# Errors report
print('Error verbose:')
show_error_report(errors_count)
print('The-End.')

