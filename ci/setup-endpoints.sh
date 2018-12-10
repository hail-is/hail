#!/bin/bash

set -e
set +x

FULLY_QUALIFIED_REPO_NAME=$1
TOKEN=$2
CALLBACK_URL=$3

if [[ -z "${FULLY_QUALIFIED_REPO_NAME}" ]] || [[ -z "${TOKEN}" ]] || [[ -z "${CALLBACK_URL}" ]]
then
    echo "USAGE: ./setup-endpoints.sh FULLY_QUALIFIED_REPO_NAME GITHUB_OAUTH_TOKEN CALLBACK_URL"
    echo "    e.g.: ./setup-endpoints.sh hail-ci-test/foo abcdef http://ci2.hail.is"
    exit 1
fi

for ENDPOINT in push pull_request pull_request_review
do
    echo "creating endpoint ${ENDPOINT}"
    echo "curl -XPOST "
    echo "     -i "
    echo "     https://api.github.com/repos/${FULLY_QUALIFIED_REPO_NAME}/hooks"
    echo "     -H \"Authorization: token REDACTED\""
    echo      -d '{ "name": "web"
             , "config": {
                 "url": "'${CALLBACK_URL}/${ENDPOINT}'"
               , "content_type": "json"
               }
             , "events": ["'${ENDPOINT}'"]
             }'
    curl -XPOST \
         -i \
         https://api.github.com/repos/${FULLY_QUALIFIED_REPO_NAME}/hooks \
         -H "Authorization: token ${TOKEN}" \
         -d '{ "name": "web"
             , "config": {
                 "url": "'${CALLBACK_URL}/${ENDPOINT}'"
               , "content_type": "json"
               }
             , "events": ["'${ENDPOINT}'"]
             }'
done

