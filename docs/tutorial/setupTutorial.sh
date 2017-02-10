#!/bin/bash

###################################################################
# Downloads tutorial files if no directory is given
# Makes symlinks if necessary
###################################################################

set -e

tutorialFiles=$1

if [ -z $tutorialFiles ]; then
    echo "wget"
    wget https://storage.googleapis.com/hail-tutorial/Hail_Tutorial_Data-v2.tgz && tar -xvzf Hail_Tutorial_Data-v2.tgz --strip 1
elif [ $tutorialFiles != `pwd` ]; then
    echo "symlink"
    for f in ${tutorialFiles}/*; do
        if [ ! -e $(basename $f) ]; then
            ln -s $f ./
        fi
    done
else
    echo "pass"
    :
fi