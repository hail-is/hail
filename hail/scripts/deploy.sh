#!/bin/bash

set -ex

retry() {
    "$@" ||
        (sleep 2 && "$@") ||
        (sleep 5 && "$@");
}

[[ $# -eq 12 ]] || (echo "./deploy.sh HAIL_PIP_VERSION HAIL_VERSION GIT_VERSION REMOTE WHEEL GITHUB_OAUTH_HEADER_FILE HAIL_GENETICS_HAIL_IMAGE HAIL_GENETICS_HAILTOP_IMAGE HAIL_GENETICS_VEP_GRCH37_85_IMAGE WHEEL_FOR_AZURE WEBSITE_TAR" ; exit 1)

HAIL_PIP_VERSION=$1
HAIL_VERSION=$2
GIT_VERSION=$3
REMOTE=$4
WHEEL=$5
GITHUB_OAUTH_HEADER_FILE=$6
HAIL_GENETICS_HAIL_IMAGE=$7
HAIL_GENETICS_HAILTOP_IMAGE=$8
HAIL_GENETICS_VEP_GRCH37_85_IMAGE=$9
WHEEL_FOR_AZURE=${10}
WEBSITE_TAR=${11}

retry skopeo inspect $HAIL_GENETICS_HAIL_IMAGE || (echo "could not pull $HAIL_GENETICS_HAIL_IMAGE" ; exit 1)
retry skopeo inspect $HAIL_GENETICS_HAILTOP_IMAGE || (echo "could not pull $HAIL_GENETICS_HAILTOP_IMAGE" ; exit 1)
retry skopeo inspect $HAIL_GENETICS_VEP_GRCH37_85_IMAGE || (echo "could not pull $HAIL_GENETICS_VEP_GRCH37_85_IMAGE" ; exit 1)

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
retry skopeo copy $HAIL_GENETICS_HAIL_IMAGE docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail:$HAIL_PIP_VERSION
retry skopeo copy $HAIL_GENETICS_HAILTOP_IMAGE docker://docker.io/hailgenetics/hailtop:$HAIL_PIP_VERSION
retry skopeo copy $HAIL_GENETICS_HAILTOP_IMAGE docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hailtop:$HAIL_PIP_VERSION
retry skopeo copy $HAIL_GENETICS_VEP_GRCH37_85_IMAGE docker://docker.io/hailgenetics/vep-grch37-85:$HAIL_PIP_VERSION
retry skopeo copy $HAIL_GENETICS_VEP_GRCH37_85_IMAGE docker://us-docker.pkg.dev/hail-vdc/hail/hailgenetics/vep-grch37-85:$HAIL_PIP_VERSION

# deploy to PyPI
twine upload $WHEEL

# deploy wheel for Azure HDInsight
wheel_for_azure_url=gs://hail-common/azure-hdinsight-wheels/$(basename $WHEEL_FOR_AZURE)
gcloud storage cp $WHEEL_FOR_AZURE $wheel_for_azure_url
gcloud storage objects update $wheel_for_azure_url --temporary-hold

# update docs sha
cloud_sha_location=gs://hail-common/builds/0.2/latest-hash/cloudtools-5-spark-2.4.0.txt
printf "$GIT_VERSION" | gcloud storage cp  - $cloud_sha_location
gcloud storage objects update -r $cloud_sha_location --add-acl-grant=entity=AllUsers,role=READER

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
