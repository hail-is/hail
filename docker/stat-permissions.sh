#!/bin/bash

set -ex

uname=$(uname -s)

if [ $uname = "Linux" ]
     stat -c "%a" $1
then
elif [ $uname = "Darwin" ]
     stat -f "%A" $1
else
    echo "unsupported OS $uname"
    exit 1
fi
