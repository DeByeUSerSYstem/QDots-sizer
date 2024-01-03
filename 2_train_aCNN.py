#!/usr/bin/python3

import os
import time
import argparse
from glob import glob
from utils import paths
from utils import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# silence tensorflow I, E and W warnings
# or any of {'0', '1', '3'} – 3 silences errors too
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.backend import clear_session
from tensorflow.compat.v1 import reset_default_graph
from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def create_encoders(all_labels):
    """
    Creates and fits encoders.
    :param all_labels: numpy-array or list of all the dataset's labels
    :return: two fitted encoders
    """
    # gather a unique list of all labels
    s = set(all_labels)
    labels = np.asarray(list(s))

    # create label encoder: transforms float-labels into integer values
    lbl_encoder = LabelEncoder()
    labels_as_int = lbl_encoder.fit_transform(labels)

    # create one-hot encoder from the integers
    onehot_encoder = OneHotEncoder(sparse_output=False)
    labels_as_int = labels_as_int.reshape(-1, 1)
    onehot_encoder.fit(labels_as_int)

    return lbl_encoder, onehot_encoder


def one_hot_encoding(x, tag, lbl_encoder=None, onehot_encoder=None):
    """
    Label encoding/decoding: floating categorical values <--> one-hot vector.
    :param x: labels to be transformed (numpy-array or list)
    :param tag: string indicating required transformation.
                Allowed values: "encode", "decode"
    :param lbl_encoder: *KEEP DEFAULT* categorical <-> integers encoder
    :param onehot_encoder: *KEEP DEFAULT* integers <-> one-hot-vector encoder
    :return: encoded or decoded list of labels
    """
    # set of permitted values for param "tag"
    valid = {'encode', 'decode'}
    if tag not in valid:
        raise ValueError('Tag parameter must be explicit. ',
                         'Accepted values: "encode", "decode".')

    # reference encoders
    if lbl_encoder is None:
        lbl_encoder = label_encoder
    if onehot_encoder is None:
        onehot_encoder = one_hot_encoder

    # encode labels
    if tag == 'encode':
        as_int = lbl_encoder.transform(x)
        as_int = as_int.reshape(-1, 1)
        encoded = onehot_encoder.transform(as_int)
        return encoded

    # decode labels
    elif tag == 'decode':
        as_int = [np.argmax(vector) for vector in x]
        original = lbl_encoder.inverse_transform(as_int)
        return original


def get_learning_curves(dataframe, round_n):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.suptitle(f'round {round_n}')

    # Loss subplot (ax1)
    ax1.plot(dataframe['loss'], color='tab:blue', marker='.', label='train')
    ax1.plot(dataframe['val_loss'], color='tab:orange', marker='.', label='val')
    ax1.set_ylim(-0.01, ax1.set_ylim()[1])  # adaptive max value
    ax1.grid(color='0.95')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right')

    # Accuracy subplot (ax2)
    ax2.plot(dataframe['accuracy'], color='tab:blue', marker='.', label='train')
    ax2.plot(dataframe['val_accuracy'], color='tab:orange', marker='.', label='val')
    ax2.set_ylim(ax2.set_ylim()[0], 1.01)  # adaptive max value
    ax2.grid(color='0.95')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_xlabel('epochs', fontsize=12)
    ax2.legend(loc='lower right')

    # Adjust figure spaces
    fig.tight_layout()


# cmaps: -Blues-, GnBu,,, PuBu, YlGn, Greens, YlOrRd (!)
def get_conf_matrix(true_val, pred_val, round_n):

    # set labels manually
    axis_labels = [str(f) for f in np.arange(2.0, 10.5, 0.5)]

    y_true = label_encoder.transform(true_val)
    y_pred = label_encoder.transform(pred_val)

    # compute the confusion matrix
    conf_mtx = confusion_matrix(y_true, y_pred, normalize='true')

    # save numerical log of the confusion matrix
    conf_mtx_log = os.path.join(reference_folder, 'Confusion_Matrix',
                                f'ConfMtx_log_{round_n}.txt')
    np.savetxt(conf_mtx_log, conf_mtx, delimiter=',', fmt='%.5f')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(conf_mtx, cmap='GnBu',
                   norm=plt_colors.LogNorm(vmin=0.01, vmax=1))

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Normalized Count', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Set tick labels and title
    ax.set_xticks(np.arange(len(axis_labels)))
    ax.set_yticks(np.arange(len(axis_labels)))
    ax.set_xticklabels(axis_labels, fontsize=10)
    ax.set_yticklabels(axis_labels, fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title(f'Confusion Matrix – Round {round_n}', fontsize=14)

    # Text annotations
    for i in range(len(axis_labels)):
        for j in range(len(axis_labels)):
            value = conf_mtx[i, j]
            text = f'{value:.2f}' if value != 0 else ''
            color = 'white' if value > 0.3 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    color=color, fontsize=10)

    # Set aspect ratio
    ax.set_aspect('equal')


def count_errors(file_name, true_val, pred_val):
    threshold = 0.5
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
    error_list = pd.DataFrame(mistakes,
                              columns=['file name', 'true size', 'prediction'])
    error_list.to_csv(os.path.join(reference_folder, 'Summary_of_predictions',
                      f'error_list_{k}.csv'), float_format='%.5f', index=False)

    return [one_class_wrong, two_class_wrong, badly_wrong, total]


def show_error_report(list_of_error_stats):
    for n, stat in enumerate(list_of_error_stats):
        one_class_err = stat[0]
        two_class_err = stat[1]
        bad_err = stat[2]
        tot = stat[3]
        one_class_err_percentage = (one_class_err / tot) * 100
        two_class_err_precentage = (two_class_err / tot) * 100
        bad_err_pergentage = (bad_err / tot) * 100
        err_sum = one_class_err + two_class_err + bad_err
        all_err_percentage = (err_sum / tot) * 100
        print()
        print(f'In round {n}:')
        print('Model was wrong {} times over {} ({:0.2f}%)'
              .format(err_sum, tot, all_err_percentage))
        if two_class_err != 0 or bad_err != 0:
            print('Of which: {} ({:0.2f}%) errors within 1 class,'
                  .format(one_class_err, one_class_err_percentage))
            print(' '*10, '{} ({:0.2f}%) within two classes,'
                  .format(two_class_err, two_class_err_precentage))
            print(' '*6, 'and {} ({:0.2f}%) huge errors'
                  .format(bad_err, bad_err_pergentage))


# Create and "error per class" histogram
def do_size_error_histograms(rep, accuracies):
    # create dataframe with all classes
    data_size = pd.DataFrame()
    data_size['classes'] = np.arange(2.0, 10.1, 0.5)
    maximum = 0

    # per each k-round:
    for r in range(rep):

        # load error list
        fname = glob(os.path.join(reference_folder, 'Summary_of_predictions',
                                  f'error_list_{r}.csv'))[0]
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
        col_name = 'round {} – Acc:{:.3f}%, err:{}/{}'.format(
                    r, accuracies[r]*100, len(error_list), len(z[test]))
        df = df.rename(columns={'occurrences': col_name})

        # add results to main df
        data_size = data_size.join(df.set_index('classes'), on='classes').fillna(0)
        maximum = df.max(axis=0)[col_name] if df.max(axis=0)[col_name] > maximum else maximum
        del df

    # save complessive error histogram
    data_size.to_csv(os.path.join(reference_folder, 'error_Histograms',
                     f'Error_counts_Size.csv'),
                     float_format='%.5f', index=False)

    # plot all columns against 'classes' index
    title = f'Error evolution per size'
    plot = data_size.plot(x='classes', kind='bar', figsize=(17, 8),
                          subplots=True, sharex=False, layout=(2, 3),
                          title=title, xlabel='classes', ylabel='error %',
                          fontsize=9, legend=False, rot=0,
                          ylim=[0, maximum+(maximum/100*5)])
    plt.tight_layout()


def do_conc_error_histograms(rep, accuracies):
    # prepare collective DataFrame
    data_conc = pd.DataFrame()
    data_conc['Concentrations'] = ['Conc100', 'Conc65', 'Conc37', 'Conc15']
    maximum = 0

    # list all needed files
    file_names = sorted(glob(os.path.join(reference_folder,
                             'Summary_of_predictions', f'error_list_*.csv')))
    total_files = len(file_names)
    assert total_files == rep

    # iterate through each file and store the corresponding frequencies
    for i, file_name in enumerate(file_names):
        # Load current CSV file
        data = pd.read_csv(file_name)
        column_data = data['file name']

        # initialize the frequency dictionary for the current file
        conc_frequency = {'Conc100': 0, 'Conc65': 0, 'Conc37': 0, 'Conc15': 0}

        # Iterate through each row of the column to check for the serach string
        for row in column_data:
            for search_string in conc_frequency.keys():
                if search_string in row:
                    conc_frequency[search_string] += 1

        # convert the frequency dictionary to a DataFrame for easier plotting
        freq_df = pd.DataFrame({'String': list(conc_frequency.keys()),
                                'conc_Frequency': list(conc_frequency.values())
                                })

        # normalise counts to 100%
        n_rows = len(column_data)
        assert n_rows == freq_df['conc_Frequency'].sum()
        freq_df['conc_Frequency'] = (freq_df['conc_Frequency'] / n_rows) * 100

        # raname column to store it in the collective DataFrame
        col_name = 'round {} – Acc:{:.3f}%, err:{}/{}'.format(
                    i, accuracies[i]*100, n_rows, len(z[test]))
        freq_df = freq_df.rename(columns={'conc_Frequency': col_name})

        # add results to main df
        data_conc = data_conc.join(freq_df.set_index('String'), on='Concentrations').fillna(0)
        maximum = freq_df.max(axis=0)[col_name] if freq_df.max(axis=0)[col_name] > maximum else maximum
        del freq_df

    # save complessive error histogram
    data_conc.to_csv(os.path.join(reference_folder, 'error_Histograms',
                     f'Error_counts_Conc.csv'),
                     float_format='%.5f', index=False)

    # plot all columns against 'Concentrations' index
    title = f'Error evolution per concentration level'
    ax = data_conc.plot(x='Concentrations', kind='bar', figsize=(12, 6),
                        subplots=True, sharex=False, layout=(2, 3),
                        title=title, xlabel='Concentrations', ylabel='error %',
                        fontsize=9, legend=False, rot=0,
                        ylim=[0, maximum+(maximum/100*5)])
    plt.tight_layout()


# --------------------------------------------
#  PGM START
# --------------------------------------------
print(time.asctime())
start = time.time()  # time check
np.set_printoptions(precision=3)

# Clear Keras and TF session, if run previously
clear_session()
reset_default_graph()

# set parameter
print('Training for Q-step= 0.5 nm, Q-range = Q14, with all solvents.')

# Set reference folder, and folders reqired for diagnostics storage
reference_folder = os.path.join(paths.data_dir, 'results')
tools.set_required_subfolders(reference_folder, 'training_logs', 'models',
                              'learning_curves', 'Confusion_Matrix',
                              'error_Histograms', 'Summary_of_predictions')


# --------------------------------------------
#  LOAD DATA AND ENCODE LABELS
# --------------------------------------------

# Load data and relative desired labels
print('=== Loading data ===')
X = np.load(os.path.join(reference_folder, 'data_matrix.npy'))
y = np.load(os.path.join(reference_folder, 'labels.npy'))
z = np.load(os.path.join(reference_folder, 'fnames.npy'))

# create encoders
label_encoder, one_hot_encoder = create_encoders(y)
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

# Run cross validation and train the a-CNN each time in loop
for k, (train, test) in enumerate(k_fold.split(X, y_hot, z)):
    b = time.time()

    # Define Training and Network parameters
    lr = 0.001
    batch_size = 256
    n_epochs = 400
    patience = 20

    # Pattern length --> Input size
    inp_len = 5004

    # Output length --> N° of classes
    n_classes = 17

    # Define Keras Model
    # https://doi.org/10.1038/s41524-019-0196-x
    model = Sequential(name='tinyCNN')

    model.add(Conv1D(32, 10, strides=10, padding='same', activation='relu',
                     input_shape=(inp_len, 1)))
    model.add(Conv1D(32, 5, strides=5, padding='same', activation='relu'))
    model.add(Conv1D(32, 4, strides=4, padding='same', activation='relu'))
    model.add(Conv1D(32, 3, strides=3, padding='same', activation='relu'))
    model.add(Conv1D(32, 2, strides=2, padding='same', activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(n_classes, activation='softmax'))

    # Print model summary (just once)
    if k == 0:
        print("\nModel summary:")
        print(model.summary(), '\n')

    # optimizer
    opt = Adam(learning_rate=lr)

    # choose early stop
    early_stop = EarlyStopping(monitor='val_loss', patience=patience,
                               restore_best_weights=True,
                               verbose=1)

    # compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit model
    print(f'\n=== Training, round {k} === {time.asctime()}')
    hist = model.fit(X[train], y_hot[train], shuffle=True,
                     batch_size=batch_size, epochs=n_epochs,
                     validation_split=0.1, callbacks=[early_stop],
                     verbose=0)

    # compute loss and accuracy for each k validation
    loss, acc = model.evaluate(X[test], y_hot[test], verbose=0)
    accuracy.append(acc)

    # compute (and decode) model predictions
    encoded_prediction = model.predict(X[test], verbose=0)
    prediction = one_hot_encoding(encoded_prediction, 'decode')

    # feedback
    c = time.time()  # time check
    print('This training round took {:.2f}min'.format((c-b)/60))


    # --------------------------------------------
    #  DIAGNOSTICS
    # --------------------------------------------

    # Save model
    model.save(os.path.join(reference_folder, 'models', f'model_{k}.h5'))

    # Save training log
    log = pd.DataFrame(hist.history)
    log.to_csv(os.path.join(reference_folder, 'training_logs', f'log_{k}.csv'),
               float_format='%.5f', index=False)

    # Plot Learning Curves: training VS eval Loss & Accuracy
    get_learning_curves(log, k)
    plt.savefig(os.path.join(reference_folder, 'learning_curves', f'lc_{k}'),
                dpi=200)
    plt.close()

    # Confusion Matrix: true VS predicted values
    assert len(test) == len(prediction)
    get_conf_matrix(y[test], prediction, k)
    plt.savefig(os.path.join(reference_folder, 'Confusion_Matrix', f'cm_{k}'),
                dpi=700)
    plt.close()

    # Save prdictions, labels and filenmes in a csv file
    all_predictions = pd.DataFrame()
    all_predictions['File names'] = z[test]
    all_predictions['True_Size'] = y[test]
    all_predictions['Predictions'] = prediction
    summary_path = os.path.join(reference_folder, 'Summary_of_predictions',
                                f'all_predictions_{k}.csv')
    all_predictions.to_csv(summary_path, float_format='%.5f', index=False)

    # Gather errors details
    errors_count.append(count_errors(z[test], y[test], prediction))


# --------------------------------------------
#  PRINT RESULTS
# --------------------------------------------

# Accuracy report
accuracy = np.asarray(accuracy)
print('\nTraining with cross-validation ended.')
print(f'Acuracies for each round are: {accuracy}')
print('--> Mean Cross-val accuracy {:.3f}% <--'.format(np.mean(accuracy)*100))

# Error histogram wrt sizes
do_size_error_histograms(folds, accuracy)
plt.savefig(os.path.join(reference_folder, 'error_Histograms',
                         f'Errror_per_Size'), dpi=200)
plt.close()

# Error histogram wrt concentrations
do_conc_error_histograms(folds, accuracy)
plt.savefig(os.path.join(reference_folder, 'error_Histograms',
                         f'Error_per_Conc'), dpi=200)
plt.close()

# feedback
print('\n', '-'*66, '\n')
print(time.asctime())
end = time.time()  # time check
print('The whole process took {:.0f}min'.format((end-start)/60))

# Error report
print('Error verbose:')
show_error_report(errors_count)
print('\nThe-End.\n\n')
