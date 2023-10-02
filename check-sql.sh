#!/bin/bash

set -e
set -o pipefail

# Verify the the working tree modifies no sql files relative to the main
# branch. This will always pass on a deploy because the working tree is an
# unmodified copy of the main branch.

target_treeish=${HAIL_TARGET_SHA:-$(git merge-base main HEAD)}

if [ ! -d sql ]; then
    echo 'No migrations to check, exiting.'
    exit 0
fi

set +e
git diff --name-status $target_treeish sql \
    | grep -Ev $'^A|^M\t[^/]+/sql/estimated-current.sql|^D\t[^/]+/sql/delete-[^ ]+-tables.sql'
grep_exit_code=$?


if [[ $grep_exit_code -eq 0 ]]
then
    echo 'At least one migration file was modified. See above.'
    exit 1
elif [[ $grep_exit_code -eq 1 ]]
then
    # Exit code 1 means nothing survived grep's filter, so no illegal changes were made
    # https://www.gnu.org/software/grep/manual/html_node/Exit-Status.html#Exit-Status-1
    exit 0
fi

exit $grep_exit_code
