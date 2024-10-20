#!/bin/bash

# This script is used to rename the images back to their original names for
# testing purposes. It is used in conjunction with the slide_extractor.py script
# rename images so that they are renamed to have the "_to_be_removed" suffix removed

# determine the path for the images based on the base name that is passed to this script as a param
base_name=$1
path_to_images="extracted_slides/testdata/$base_name"

# We also want to remain in our current directory, so we need to store the current directory
current_dir=$(pwd)

# now rename the images back to their original names (ie remove the "_to_be_removed_pass_n" suffix), where n is a number
# example: slide_001_to_be_removed_pass_1.png -> slide_001.png
cd $path_to_images

for file in *; do
    if [[ $file == *_to_be_removed_pass_* ]]; then
        new_name=$(echo $file | sed 's/_to_be_removed_pass_[0-9]*//')
        mv $file $new_name
    fi
done

# return to the original directory
cd $current_dir

echo "Done renaming images back to their original names!"

