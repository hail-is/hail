#!/bin/bash

set -e

uname=$(uname -s)

if [ $uname = "Linux" ]
then
     stat -c "%a" $1
elif [ $uname = "Darwin" ]
then
     stat -f "%A" $1
else
    echo "unsupported OS $uname"
    exit 1
fi
