#!/bin/bash

# check if environment variable EXAM is set
if [ -n "$EXAM" ]; then

    if [ -n "$FORCE" ]; then
        git fetch
        git reset --hard origin/exam
    else
        git checkout exam
        git fetch
        git pull
    fi

    tar_files=(*.tar.gz)
    echo "Select a file to extract:"
    select tar_file in "${tar_files[@]}"; do
        if [ -n "$tar_file" ]; then
            break
        else
            echo "Invalid selection. Please try again."
        fi
    done

    openssl enc -aes-256-cbc -pbkdf2 -iter 100000 -d -in "$tar_file" -pass pass:"$EXAM" | tar -xz -C .

    # if that is successful, remove the tar file
    if [ $? -eq 0 ]; then
        rm "$tar_file"
    else
        echo "Failed to extract the tar file. Please check your password and try again."
    fi

    # download slides
    echo "Downloading extra material..."
    curl -o ML_exam_slides.tar.xz https://cloud.dei.unipd.it/public.php/dav/files/qiiccDGqZREDbHB

    # check if the download was successful
    if [ $? -ne 0 ]; then
        echo "Failed to download slides. Please check your internet connection."
    else
        tar -xf ML_exam_slides.tar.xz
        rm ML_exam_slides.tar.xz
    fi
fi

# start Jupyter Lab
jupyter lab
