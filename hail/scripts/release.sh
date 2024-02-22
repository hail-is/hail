#!/bin/bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

usage() {
cat << EOF
usage: $(basename "$0")

    All arguments are specified by environment variables. For example:

        HAIL_PIP_VERSION=0.2.123
        HAIL_VERSION=0.2.123-abcdef123
        GIT_VERSION=abcdef123
        REMOTE=origin
        WHEEL=/path/to/the.whl
        GITHUB_OAUTH_HEADER_FILE=/path/to/github/oauth/header/file
        HAIL_GENETICS_HAIL_IMAGE=docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-123abc
        HAIL_GENETICS_HAIL_IMAGE_PY_3_10=docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-123abc
        HAIL_GENETICS_HAIL_IMAGE_PY_3_11=docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:deploy-123abc
        HAIL_GENETICS_HAILTOP_IMAGE=docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hailtop:deploy-123abc
        HAIL_GENETICS_VEP_GRCH37_85_IMAGE=docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep-grch37-85:deploy-123abc
        HAIL_GENETICS_VEP_GRCH38_95_IMAGE=docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep-grch38-95:deploy-123abc
        AZURE_WHEEL=/path/to/wheel/for/azure
        WEBSITE_TAR=/path/to/www.tar.gz
        $(basename "$0")
EOF
}

retry() {
    "$@" ||
        (sleep 2 && "$@") ||
        (sleep 5 && "$@");
}

arguments="HAIL_PIP_VERSION HAIL_VERSION GIT_VERSION REMOTE WHEEL GITHUB_OAUTH_HEADER_FILE \
           HAIL_GENETICS_HAIL_IMAGE HAIL_GENETICS_HAIL_IMAGE_PY_3_10 \
           HAIL_GENETICS_HAIL_IMAGE_PY_3_11 HAIL_GENETICS_HAILTOP_IMAGE \
           HAIL_GENETICS_VEP_GRCH37_85_IMAGE HAIL_GENETICS_VEP_GRCH38_95_IMAGE AZURE_WHEEL \
           WEBSITE_TAR"

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

retry skopeo inspect $HAIL_GENETICS_HAIL_IMAGE || (echo "could not pull $HAIL_GENETICS_HAIL_IMAGE" ; exit 1)
retry skopeo inspect $HAIL_GENETICS_HAIL_IMAGE_PY_3_10 || (echo "could not pull $HAIL_GENETICS_HAIL_IMAGE_PY_3_10" ; exit 1)
retry skopeo inspect $HAIL_GENETICS_HAIL_IMAGE_PY_3_11 || (echo "could not pull $HAIL_GENETICS_HAIL_IMAGE_PY_3_11" ; exit 1)
retry skopeo inspect $HAIL_GENETICS_HAILTOP_IMAGE || (echo "could not pull $HAIL_GENETICS_HAILTOP_IMAGE" ; exit 1)
retry skopeo inspect $HAIL_GENETICS_VEP_GRCH37_85_IMAGE || (echo "could not pull $HAIL_GENETICS_VEP_GRCH37_85_IMAGE" ; exit 1)
retry skopeo inspect $HAIL_GENETICS_VEP_GRCH38_95_IMAGE || (echo "could not pull $HAIL_GENETICS_VEP_GRCH38_95_IMAGE" ; exit 1)

if git ls-remote --exit-code --tags $REMOTE $HAIL_PIP_VERSION
then
    echo "tag $HAIL_PIP_VERSION already exists"
    exit 0
fi

if [ ! -f $WHEEL ]
then
    echo "wheel not found at $WHEEL"
    exit 1
fi

pip_versions_file=$(mktemp)
pip install hail== 2>&1 \
    | head -n 1 \
    | sed 's/.*versions: //' \
    | sed 's/)//' \
    | sed 's/ //g' \
    | tr ',' '\n' \
         > $pip_versions_file

if grep -q -e $HAIL_PIP_VERSION $pip_versions_file
then
    echo "package $HAIL_PIP_VERSION already exists"
    exit 1
fi

if curl -sf https://api.github.com/repos/hail-is/hail/releases/tags/$HAIL_PIP_VERSION >/dev/null
then
    echo "release $HAIL_PIP_VERSION already exists"
    exit 1
fi

export PYPI_USED_STORAGE=$(curl https://pypi.org/pypi/hail/json | jq '[.releases[][].size ]| add')
WHEEL=$WHEEL python3 $SCRIPT_DIR/assert_pypi_has_room.py

# push git tag
git tag $HAIL_PIP_VERSION -m "Hail version $HAIL_PIP_VERSION."
git push origin $HAIL_PIP_VERSION

# make GitHub release
curl -XPOST -H @$GITHUB_OAUTH_HEADER_FILE https://api.github.com/repos/hail-is/hail/releases -d '{
  "tag_name": "'$HAIL_PIP_VERSION'",
  "target_commitish": "main",
  "name": "'$HAIL_PIP_VERSION'",
  "body": "Hail version '$HAIL_PIP_VERSION'\n\n[Change log](https://hail.is/docs/0.2/change_log.html#version-'${HAIL_PIP_VERSION//[\.]/-}')",
  "draft": false,
  "prerelease": false
}'

retry skopeo copy $HAIL_GENETICS_HAIL_IMAGE docker://docker.io/hailgenetics/hail:$HAIL_PIP_VERSION
retry skopeo copy $HAIL_GENETICS_HAIL_IMAGE docker://docker.io/hailgenetics/hail:$HAIL_PIP_VERSION-py3.9
retry skopeo copy $HAIL_GENETICS_HAIL_IMAGE docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:$HAIL_PIP_VERSION
retry skopeo copy $HAIL_GENETICS_HAIL_IMAGE docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:$HAIL_PIP_VERSION-py3.9

retry skopeo copy $HAIL_GENETICS_HAIL_IMAGE_PY_3_10 docker://docker.io/hailgenetics/hail:$HAIL_PIP_VERSION-py3.10
retry skopeo copy $HAIL_GENETICS_HAIL_IMAGE_PY_3_10 docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:$HAIL_PIP_VERSION-py3.10

retry skopeo copy $HAIL_GENETICS_HAIL_IMAGE_PY_3_11 docker://docker.io/hailgenetics/hail:$HAIL_PIP_VERSION-py3.11
retry skopeo copy $HAIL_GENETICS_HAIL_IMAGE_PY_3_11 docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:$HAIL_PIP_VERSION-py3.11

retry skopeo copy $HAIL_GENETICS_HAILTOP_IMAGE docker://docker.io/hailgenetics/hailtop:$HAIL_PIP_VERSION
retry skopeo copy $HAIL_GENETICS_HAILTOP_IMAGE docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hailtop:$HAIL_PIP_VERSION

retry skopeo copy $HAIL_GENETICS_VEP_GRCH37_85_IMAGE docker://docker.io/hailgenetics/vep-grch37-85:$HAIL_PIP_VERSION
retry skopeo copy $HAIL_GENETICS_VEP_GRCH37_85_IMAGE docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep-grch37-85:$HAIL_PIP_VERSION

retry skopeo copy $HAIL_GENETICS_VEP_GRCH38_95_IMAGE docker://docker.io/hailgenetics/vep-grch38-95:$HAIL_PIP_VERSION
retry skopeo copy $HAIL_GENETICS_VEP_GRCH38_95_IMAGE docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep/grch38-95:$HAIL_PIP_VERSION


# deploy to PyPI
twine upload $WHEEL

# deploy wheel for Azure HDInsight
azure_wheel_url=gs://hail-common/azure-hdinsight-wheels/$(basename $AZURE_WHEEL)
gcloud storage cp $AZURE_WHEEL $azure_wheel_url
gcloud storage objects update $azure_wheel_url --temporary-hold

# deploy datasets (annotation db) json
datasets_json_url=gs://hail-common/annotationdb/$HAIL_VERSION/datasets.json
gcloud storage cp python/hail/experimental/datasets.json $datasets_json_url
gcloud storage objects update $datasets_json_url --temporary-hold

# Publish website
website_url=gs://hail-common/website/$HAIL_PIP_VERSION/www.tar.gz
gcloud storage cp $WEBSITE_TAR $website_url
gcloud storage objects update $website_url --temporary-hold

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
