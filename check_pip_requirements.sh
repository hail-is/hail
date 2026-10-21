#!/bin/bash

set -ex

for package in $@;
do
    reqs="$package/requirements.txt"
    pinned="$package/pinned-requirements.txt"
    new_pinned=$(mktemp)
    pinned_no_comments=$(mktemp)
    new_pinned_no_comments=$(mktemp)

    uv pip compile \
        --python-version 3.12 \
        --python-platform linux \
        --generate-hashes \
        --quiet $reqs $pinned --output-file=$new_pinned
    # Compare versions only. Comments can differ despite requirements being the same, and the
    # hash list for a pinned version grows if PyPI later gains new files for it (pip checks
    # hashes at install time).
    sed -e '/#/d' -e '/--hash=/d' -e 's/[[:space:]]*\\$//' $pinned > $pinned_no_comments
    sed -e '/#/d' -e '/--hash=/d' -e 's/[[:space:]]*\\$//' $new_pinned > $new_pinned_no_comments
    diff $pinned_no_comments $new_pinned_no_comments || {
        echo '>>> up-to-date pinned requirements <<<'
        cat $new_pinned
        echo '--------------------------------------'
        echo "$pinned is no longer up to date with $reqs"
        echo "Please regenerate the pinned requirements file by running:"
        echo "> make $pinned"
        exit 1
    }
done
