#!/bin/bash

# Check if a directory path is provided as an argument
if [ $# -eq 0 ]; then
    echo "Please provide a directory path as an argument."
    exit 1
fi

# Directory path provided as an argument
folder_path="$1"

# Check if the specified path exists and is a directory
if [ ! -d "$folder_path" ]; then
    echo "The specified path is not a directory or doesn't exist."
    exit 1
fi

# Change to the specified directory
cd "$folder_path" || exit 1

# Extract all .tar.gz files in the directory
for file in *.tar.gz; do
    if [ -f "$file" ]; then
        echo "Extracting $file..."
        tar -xzf "$file"
    fi
done

echo "Extraction completed for all .tar.gz files in the directory."