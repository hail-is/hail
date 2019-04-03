#!/bin/sh

op=$1

while read user; do
    make user name="$user" op=$op
done <users.txt