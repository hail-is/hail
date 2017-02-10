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

if [ -z $tutorialFiles ]; then
    wget https://storage.googleapis.com/hail-tutorial/Hail_Tutorial_Data-v2.tgz && tar -xvzf Hail_Tutorial_Data-v2.tgz --strip 1
else
    for f in ${tutorialFiles}/*; do
        if [ ! -e $(basename $f) ]; then
            ln -s $f ./
        fi
    done
fi