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

if git diff --name-status --diff-filter=r $target_treeish sql \
    | grep -Ev $'^A|^M\t[^/]+/sql/estimated-current.sql|^D\t[^/]+/sql/delete-[^ ]+-tables.sql'
then
    echo 'At least one migration file was modified. See above.'
    exit 1
fi
# NOTE: Exit code 1 (above if failed) means nothing survived grep's filter, so
# no illegal changes were made. See link for more details:
#     https://www.gnu.org/software/grep/manual/html_node/Exit-Status.html#Exit-Status-1
