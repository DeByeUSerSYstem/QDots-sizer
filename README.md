# QDots-sizer

This is a python package for automated size classification of semiconductor Quantum Dots
from Wide-Angle x-ray Total Scattering pattern, using a light all-convolutional deep
learning model trained with physics-informed augmented data.


### Supported data type
 * Sample state: colloidal and dry
 * Angular range: wide (synchrotron-like) and narrow (laboratory machine-like)


## Installation
To install, just clone the current repository:
```
git clone https://github.com/DeByeUSerSYstem/QDots-sizer.git
```


## Pre-requisites
In order to run the program, you need to have installed:

 * Python 3.8 or higher
 * pip version 19.0 or higher (limit imposed by TensorFlow intallation)
 * numpy
 * matplotlib
 * TensorFlow 2 version 2.11.0 or higher (to back Keras module) (\*)

If needed, the last three can be installed via `python -m pip install numpy matplotlib tensorflow`  (\*)

(\*) Please note that both CPU and GPU installation are available (and supported), but
have different installation procedures. Refer to the official TensorFlow documentation
<a href="https://www.tensorflow.org/install">here</a>.


## Usage
A pre-trained model for PbS is available in the repository. To classify any pattern
of this material, go into the folder where you cloned the repository and run the following:
```
python Predict_datum_by_name.py path_to_diffractogram -l collection_wavelength
```
The process should take just a few seconds. Both CPU and GPU can be used, depending
on the TensorFlow version you have installed.

At the end of the procedure, you will find a new file.xy with the predicted size in the file name and
containing your pattern as the the algorithm sees it. You will also be presented with the option 
of saving tahe plot of this.

_Argument_<br/>
`path_to_diffractogram`: compulsory<br/>
Relative of Full path of the diffractogram you wish to have a size prediction on.<br/>
Data have to be in a two-columns xy file structure: the first column containing 2θ angles,
the second column the measured intensity.

_Flags_<br/>
`-l x.xxx`: optional<br/>
Wavelength used during data collection, in Å<br/> 
If omitted, the algorithm will use the Kα1-Cu lambrda = 1.5406 Å by default.

`--header_rows n`: optional<br/>
Number of header lines at the beginning of your data to be skipped.<br/> 
If omitted, all lines will be read.<br/>
Any line beginning with `#` is automatically skipped by the program.

`--help`<br/>
Shows the automatic help topic.

### Suggestions and limits on the data 
The model 

_section under preparation_
tipo Qmin 4, 2th sarebbe 70?
noise mglio se basso
solventi?
...


## Attribution
*Authors:*
Lucia Allara <a href="https://github.com/goatworks">@goatworks</a>,
Federica Bertolotti <a href="https://github.com/febertol">@febertol</a>,
and Antonella Guagliardi.

*Version:* 1.0 / October, 2023

*e-mail of repo owner*: lallara@uninsubria.it, lucia.allara@gmail.com<br/>
*e-mail of contributors*: federica.bertolotti@uninsubria.it, antonella.guagliardi@ic.cnr.it

*License:* GPL-3.0 – Please, acknowledge use of this work with the appropriate citation.

*Citation:* you can find it on the righ bar under the *"Cite this repository"* section.<br/>
*Preprint on ChemRxiv*: https://doi.org/10.26434/chemrxiv-2023-127s9
