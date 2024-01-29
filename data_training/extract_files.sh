#!/bin/bash

# Output directory path provided as an argument
output_directory="dataset_extracted"
mkdir $output_directory

# Extract all .tar.gz files in the directory
echo "Extracting files in $output_directory..."
for file in *.tar.gz; do
    if [ -f "$file" ]; then
        echo "Extracting $file"
        tar -xzf "$file" -C "$output_directory"
    fi
done

echo "Extraction completed for all .tar.gz files."
echo ""

echo "Creating data matrices..."
python do_data_matrices.py $output_directory

echo ""
echo "Ready for training."
echo ""