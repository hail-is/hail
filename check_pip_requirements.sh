#!/bin/bash

set -ex

for package in $@;
do
    reqs="$package/requirements.txt"
    pinned="$package/pinned-requirements.txt"
    new_pinned=$(mktemp)
    pinned_no_comments=$(mktemp)
    new_pinned_no_comments=$(mktemp)

    PATH="$PATH:$HOME/.local/bin" pip-compile --quiet $reqs $pinned --output-file=$new_pinned
    # Get rid of comments that might differ despite requirements being the same
    cat $pinned | sed '/#/d' > $pinned_no_comments
    cat $new_pinned | sed '/#/d' > $new_pinned_no_comments
    diff $pinned_no_comments $new_pinned_no_comments || {
        echo '>>> up-to-date pinned requirements <<<'
        cat $new_pinned
        echo '--------------------------------------'
        echo "$pinned is no longer up to date with $reqs"
        echo "Please regenerate the pinned requirements file."
        exit 1
    }
done
