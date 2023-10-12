import os
import time
import shutil
import argparse
import numpy as np



def initialise_folder(folder_name):
    """
    Creates a directory (or nested tree). Empty it (them), if already exist.
    :param folder_name: name of the directory
    :return: nothing, acts on folder
    """
    short_name = folder_name[24:]

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(f'\nWARNING! Emptying \"{short_name}\" folder: 10 seconds to abort...')
        time.sleep(11)  # sleeps for 11 s
        shutil.rmtree(folder_name)
        os.makedirs(folder_name)


def set_required_subfolders(parent, *args):
    """
    Creates one or more directories (or nested tree).
    :param *args: (list of) name(s) of the directory
    :return: nothing, acts on folder
    """
    for folder_name in args:
        new_folder = os.path.join(parent, folder_name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)


# Q = (4π/λ) * sin(2Theta/2)
def tt2q(angle, wavelength):
    """
    Converts a 2Theta angle in the equivalent Q value.
    :param angle: the 2Theta angle – float
    :param wavelength: wavelength(Å) used during data collection – float
    :return: Q value correspondent to the given 2Theta angle (:param angle:) – float
    """
    return (4 * np.pi / wavelength) * np.sin(np.radians(angle / 2))


# 2θ = 2*arcsin(sinθ) where sinθ = Q*λ/4π
def q2tt(q, wavelength):
    """
    Converts a Q value in the equivalent 2Theta angle.
    :param q: Q value – float
    :param wavelength: wavelength(Å) used during data collection – float
    :return: 2Theta angle corresponding to Q (:param q:) – float
    """
    sinTheta = (q * wavelength) / (4 * np.pi)
    angle = 2 * np.rad2deg( np.arcsin(sinTheta) )
    return angle

