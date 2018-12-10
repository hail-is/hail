#!/bin/sh

TT_NODE="node ./index.js"

# NODEJS Watcher
if [ -z `pgrep -f -x "$TT_NODE"` ] 
then
    echo "Starting $TT_NODE."
    cmdNODE="$TT_NODE >> ./logs/node.log &"
    eval $cmdNODE
fi

TT_NODE2="perl ../../Scripts/QueryAnnotation10_sock_multi_angular.pl"

# NODEJS Watcher
if [ -z `pgrep -f -x "$TT_NODE2"` ] 
then
    echo "Starting $TT_NODE2."
    cmdNODE2="$TT_NODE2 >> ./logs/QueryAnnotation10_sock.log &"
    eval $cmdNODE2
fi