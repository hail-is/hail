#!/bin/bash

# Verify the the working tree modifices no sql files relative to the main
# branch. This will always pass on a deploy because the working tree is an
# unmodified copy of the main branch.

target_treeish=${HAIL_TARGET_SHA:-$(git merge-base main HEAD)}

modified_sql_file_list=$(mktemp)

if [ ! -d sql ]; then
    echo 'No migrations to check, exiting.'
    exit 0
fi

git diff --name-status $target_treeish sql \
    | grep -Ev $'^A|^M\t[^/]+/sql/(estimated-current.sql|delete-[^ ]+-tables.sql)' \
           > $modified_sql_file_list

if [ "$(cat $modified_sql_file_list | wc -l)" -ne 0 ]
then
    cat $modified_sql_file_list
    echo 'At least one migration file was modified. See above.'
    exit 1
fi
