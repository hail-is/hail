#!/bin/sh

set -x

echo SPARK_HOME
echo $SPARK_HOME
echo HAIL_HOME
echo $HAIL_HOME
echo PYTHONPATH
echo $PYTHONPATH
echo PYSPARK_SUBMIT_ARGS
echo $PYSPARK_SUBMIT_ARGS
echo CXX
echo $CXX
which pyspark
which pyspark2
which python
which pip

which gcc
gcc --version
which clang
clang --version

uname -a

if python -V 2>&1 | grep -q '3.[6789]'
then
    PYTHON=python
else
    python -V
    if python3 -V 2>&1 | grep -q '3.[6789]'
    then
        python3 -V
        PYTHON=python3
    else
        exit
    fi
fi

$PYTHON -m pip show hail
$PYTHON -m pip show pyspark
$PYTHON -c 'import hail as hl; hl.balding_nichols_model(3, 100, 100)._force_count_rows()'
$PYTHON -c 'import numpy as np; np.__config__.show()'

if [ ! -z "$1" ]
then
    logfile=$(ls -rt hail-*.log | tail -n 1)
    if [ ! -z "$logfile" ]
    then
        cat $logfile \
            | curl -s \
                   -F "expiry_days=10" \
                   -F "content=<-" \
                   -F "title=$logfile" \
                   http://dpaste.com/api/v2/
    fi

    corefile=$(ls -rt *hs_err_pid* | tail -n 1)
    if [ ! -z "$corefile" ]
    then
        cat $corefile \
            | curl -s \
                   -F "expiry_days=10" \
                   -F "content=<-" \
                   -F "title=$corefile" \
                   http://dpaste.com/api/v2/
    fi
fi
