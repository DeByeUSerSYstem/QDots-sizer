# QDots-sizer

This is a python package for automated size classification of quantum dots (with average diameter within 10 nm)
from  x-ray scattering data, using a light all-convolutional deep
learning model trained with physics-informed augmented data.


### Supported data type
 * Sample state: colloidal and dry
 * Angular range: custom


## Installation
### This package
Just clone the current repository by typing in a terminal:
```
git clone https://github.com/DeByeUSerSYstem/QDots-sizer.git
```
### noisy
Please download this utility from `https://github.com/SineBell/noisy` and follow the relative installation guide.


## Pre-requisites
In order to run the program, you need to have installed:

 * Python (version 3.8 or higher) and pip (version 19.0 or higher) – limits imposed by TensorFlow installation
 * Python modules: numpy, pandas, matplotlib, scikit-learn
 * TensorFlow: version 2.11.0 or higher (as backing for the Keras module) (\*)

If needed, the modules named in the last two points can be installed via `python -m pip install -U module_name`

(\*) Please note that both CPU and GPU installation are available (and supported), but
have different installation procedures. Refer to the official TensorFlow documentation
<a href="https://www.tensorflow.org/install">here</a>.


## Usage

### Data Prediction
A pre-trained model for PbS is available in the repository. To classify any pattern
of this material, go into the folder where you cloned the repository and run the following:
```
python Predict_datum_by_name.py path_to_diffractogram -l wavelength
```
The process should take just a few seconds. Both CPU and GPU can be used, depending
on the TensorFlow version you have installed.

At the end of the procedure, you will find a new file.xy with the predicted size in the file name and
containing your pattern as the the algorithm sees it. You will also be presented with the option 
of saving a plot of this pattern.

_Argument_<br/>
`path_to_diffractogram`: mandatory<br/>
Relative of Full path of the diffractogram you wish to have a size prediction on.<br/>
Data have to be in a two-columns xy file structure: the first column containing 2θ angles,
the second column the measured intensity.

_Flags_<br/>
`-l wavelength`: optional<br/>
Wavelength used during data collection, in Å<br/> 
If omitted, the algorithm will use the Kα1-Cu (λ = 1.5406 Å) by default.

`--header_rows N`: optional<br/>
Number of header lines at the beginning of your data to be skipped.<br/> 
If omitted, all lines will be read.<br/>
Any line beginning with `#` is automatically skipped by the program.

`--help`<br/>
Shows the automatic help topic.

### Data Augmentation and Network Training
Two separate scripts can be found, one for each task.
Within each script, one can find the various augmentation and training parameters, 
which can be modified by the user if needed. These parameters include, but are not limited
to: nanocrystals/solvent ratio and noise levels, Q-range limits, training speed and duration, CNN architecture.

If you wish to create your own database and train your own model, just open a terminal, 
go to the folder where this repository has been cloned and type:
```
python 1_augment_database.py path_to_your_simulation_folder && python 2_train_aCNN.py
```
where `path_to_your_simulation_folder` is the path to the folder containing your initial diffractograms
that will be augmented and then used as dataset for the model trainig. 
We recommend to simulate these initial diffractograms using the dedicated software
<a href="https://debyeusersystem.github.io/">_Debussy_</a>, 
developed in our To.Sca.Lab group @University of Insubria (Italy).

If, instead, you wish to use the already augmented dataset we have made availble, then open a terminal and type:
* `cd data_training/`: to go into the _data_training_ folder (a more detailed README can be found here, if you wish);
* `sh extract_files.sh`: this will extract all the diffractogram, making them ready for the training sctipt to use;
* `cd ..`: to go back one level, to the main project folder;
* `python 2_train_aCNN.py`: to train the algorithm.


## Attribution
*Authors:*
Lucia Allara <a href="https://github.com/goatworks">@goatworks</a>,
Federica Bertolotti <a href="https://github.com/febertol">@febertol</a>,
and Antonella Guagliardi.

*Version:* 1.0 / October, 2023

*repo owner e-mail*: lallara@uninsubria.it, lucia.allara@gmail.com<br/>
*contributors e-mail*: federica.bertolotti@uninsubria.it, antonella.guagliardi@ic.cnr.it

*License:* GPL-3.0 – Please, acknowledge use of this work with the appropriate citation.

*Citation:* you can find it on the righ bar under the *"Cite this repository"* section.<br/>
*Full paper OpenAccess*: https://doi.org/10.1038/s41524-024-01241-6
