#!/bin/bash

###################################################################
# Downloads tutorial files if no directory is given
# Makes symlinks if necessary
###################################################################

set -e

tutorialFiles=$1

if [ -d 1kg.vds/ ]; then
    rm -r 1kg.vds
fi

if [ -z "$tutorialFiles" ]; then
    if command -v wget >/dev/null 2>&1
    then
        wget https://storage.googleapis.com/hail-tutorial/Hail_Tutorial_Data-v2.tgz
    elif command -v curl >/dev/null 2>&1
    then
        curl -LO https://storage.googleapis.com/hail-tutorial/Hail_Tutorial_Data-v2.tgz
    else
        echo "building the Hail documentation requires either wget or curl" 1>&2
        exit 1
    fi
    
    tar -xvzf Hail_Tutorial_Data-v2.tgz --strip 1
else
    for f in "${tutorialFiles}"/*; do
        if [ ! -e $(basename "$f") ]; then
            ln -s "$f" ./
        fi
    done
fi

rm -rf .ipynb_checkpoints
