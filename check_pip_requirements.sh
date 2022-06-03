#!/bin/bash

set -ex

reqs=$1
pinned=$2
new_pinned=$(mktemp)
pinned_no_comments=$(mktemp)
new_pinned_no_comments=$(mktemp)

pip-compile --quiet $reqs $pinned --output-file=$new_pinned
# Get rid of comments that might differ despite requirements being the same
cat $pinned | sed '/#/d' > $pinned_no_comments
cat $new_pinned | sed '/#/d' > $new_pinned_no_comments
diff $pinned_no_comments $new_pinned_no_comments
