#!/usr/bin/env bash

set -ex

usage() {
cat << EOF
usage: $(basename "$0")

    All arguments are specified by environment variables. For example:

        HAIL_PIP_VERSION=0.2.123
        GITHUB_OAUTH_HEADER_FILE=/path/to/github/oauth/header/file
        bash $(basename "$0")
EOF
}

arguments="HAIL_PIP_VERSION GITHUB_OAUTH_HEADER_FILE"

for varname in $arguments
do
    if [ -z "${!varname}" ]  # A bash-ism, but we are #!/bin/bash
    then
        echo
        usage
        echo
        echo "$varname is unset or empty"
        exit 1
    else
        echo "$varname=${!varname}"
    fi
done

# Create pull request to update Terra and AoU Hail versions
terra_docker_dir=$(mktemp -d)
update_terra_image_py="$(cd "$(dirname "$0")" && pwd)/update-terra-image.py"
git clone https://github.com/DataBiosphere/terra-docker $terra_docker_dir
pushd $terra_docker_dir
git config user.name hail
git config user.email hail@broadinstitute.org

make_pr_for() {
    branch_name=update-$1-to-hail-$HAIL_PIP_VERSION
    git checkout -B $branch_name
    python3 $update_terra_image_py $HAIL_PIP_VERSION $1
    git commit -m "Update $1 to Hail version $HAIL_PIP_VERSION" -- config/conf.json $1
    git push -f origin HEAD
    echo "{
  \"head\": \"$branch_name\",
  \"base\": \"master\",
  \"title\": \"Update $1 to Hail $HAIL_PIP_VERSION\"
}"
    curl -XPOST -H @$GITHUB_OAUTH_HEADER_FILE https://api.github.com/repos/DataBiosphere/terra-docker/pulls -d "{
  \"head\": \"$branch_name\",
  \"base\": \"master\",
  \"title\": \"Update $1 to Hail $HAIL_PIP_VERSION\"
}"
    git reset --hard HEAD
    git checkout master
}

make_pr_for terra-jupyter-hail
make_pr_for terra-jupyter-aou
