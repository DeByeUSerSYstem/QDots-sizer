### Training set
In this folder the full dataset of Debussy simulation + physics based augmentation used for training/validation/testing of the CNN is provided.</br>
Due to GitHub limitations on the size of the files, these have been grouped in 51 folder and compressed as tar.gz files. </br>
To extract all tar.gz in a folder please launch the bash script provided, as follows:</br>
```
sh extract_files.sh
```  
This will extract the files in the folder `dataset_extracted` and create the .npy files needed by the training algorithm in `QDots_sizer/DATA/file.npy`

N.B.: Once the files have beed extracted be aware of the large size of the generated folder (about 8 GB).
