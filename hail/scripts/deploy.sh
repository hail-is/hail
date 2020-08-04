#!/bin/bash

set -ex

[[ $1 ]] || (echo "./deploy.sh HAIL_PIP_VERSION HAIL_VERSION GIT_VERSION REMOTE WHEEL GITHUB_OAUTH_HEADER_FILE" ; exit 1)
[[ $2 ]] || (echo "./deploy.sh HAIL_PIP_VERSION HAIL_VERSION GIT_VERSION REMOTE WHEEL GITHUB_OAUTH_HEADER_FILE" ; exit 1)
[[ $3 ]] || (echo "./deploy.sh HAIL_PIP_VERSION HAIL_VERSION GIT_VERSION REMOTE WHEEL GITHUB_OAUTH_HEADER_FILE" ; exit 1)
git cat-file -e $3^{commit} || (echo "bad sha $3" ; exit 1)
[[ $4 ]] || (echo "./deploy.sh HAIL_PIP_VERSION HAIL_VERSION GIT_VERSION REMOTE WHEEL GITHUB_OAUTH_HEADER_FILE" ; exit 1)
[[ $5 ]] || (echo "./deploy.sh HAIL_PIP_VERSION HAIL_VERSION GIT_VERSION REMOTE WHEEL GITHUB_OAUTH_HEADER_FILE" ; exit 1)
[[ $6 ]] || (echo "./deploy.sh HAIL_PIP_VERSION HAIL_VERSION GIT_VERSION REMOTE WHEEL GITHUB_OAUTH_HEADER_FILE" ; exit 1)

HAIL_PIP_VERSION=$1
HAIL_VERSION=$2
GIT_VERSION=$3
REMOTE=$4
WHEEL=$5
GITHUB_OAUTH_HEADER_FILE=$6

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

if curl -sf https://github.com/hail-is/hail/releases/tag/$HAIL_PIP_VERSION >/dev/null
then
    echo "release $HAIL_PIP_VERSION already exists"
    exit 1
fi

docs_location=gs://hail-common/builds/0.2/docs/hail-0.2-docs-$GIT_VERSION.tar.gz

if ! gsutil ls $docs_location
then
    echo "docs for $GIT_VERSION do not exist"
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
  "body": "Hail version '$HAIL_PIP_VERSION'",
  "draft": false,
  "prerelease": false
}'

# deploy to PyPI
twine upload $WHEEL

# update docs sha
cloud_sha_location=gs://hail-common/builds/0.2/latest-hash/cloudtools-5-spark-2.4.0.txt
printf "$GIT_VERSION" | gsutil cp  - $cloud_sha_location
gsutil acl set public-read $cloud_sha_location

# deploy annotation db json
annotation_db_json_url=gs://hail-common/annotationdb/$HAIL_VERSION/annotation_db.json
gsutil cp python/hail/experimental/annotation_db.json $annotation_db_json_url
gsutil -m retention temp set $annotation_db_json_url
