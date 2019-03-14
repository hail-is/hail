#!/bin/bash
set -ex

export UUID=${UUID:-$(../generate-uid.sh)}
export REPO_NAME=ci-test-$UUID
export WATCHED_TARGETS='[["hail-ci-test/'${REPO_NAME}':master", true]]'

set +x
TOKEN=$(cat github-tokens/user1)
set -x

cleanup() {
    trap "" INT TERM
    set +e

    for table in ${tables[@]}; do
        python3 -c "from batch.database import Database; db = Database.create_synchronous(\"$CLOUD_SQL_CONFIG_PATH\"); db.drop_table_sync(\"$table\")"
    done    
    
    [[ -z $batch_pid ]] || (kill $batch_pid; kill -9 $batch_pid)
    [[ -z $ci_pid ]] || (kill $ci_pid; kill -9 $ci_pid)
    [[ -z $proxy_pid ]] || (kill $proxy_pid; kill -9 $proxy_pid)
    set +x
    curl -XDELETE \
         -i \
         https://api.github.com/repos/hail-ci-test/${REPO_NAME} \
         -H "Authorization: token ${TOKEN}"
    set -x
}
trap cleanup EXIT
trap "exit 24" INT TERM

python3 -m pip install --user -U ../batch

if [[ -z $IN_HAIL_CI ]]; then
    export CLOUD_SQL_CONFIG_PATH=`pwd`/batch-secrets/batch-test-cloud-sql-config.json
    connection_name=$(jq -r '.connection_name' $CLOUD_SQL_CONFIG_PATH)
    host=$(jq -r '.host' $CLOUD_SQL_CONFIG_PATH)
    port=$(jq -r '.port' $CLOUD_SQL_CONFIG_PATH)
    ./cloud_sql_proxy -instances=$connection_name=tcp:$port &
    proxy_pid=$!
    ../until-with-fuel 30 curl -fL $host:$port
else
    export CLOUD_SQL_CONFIG_PATH=/batch-secrets/batch-test-cloud-sql-config.json
fi

export JOBS_TABLE=jobs-$(../generate-uid.sh)
export JOBS_PARENTS_TABLE=jobs-parents-$(../generate-uid.sh)
export BATCH_TABLE=batch-$(../generate-uid.sh)
export BATCH_JOBS_TABLE=batch-jobs-$(../generate-uid.sh)
tables=($JOBS_TABLE $JOBS_PARENTS_TABLE $BATCH_TABLE $BATCH_JOBS_TABLE)

export BATCH_SERVER_URL=http://127.0.0.1:5001
python3 -c 'import batch.server; batch.server.serve(5001)' & batch_pid=$!

../until-with-fuel 30 curl -fL 127.0.0.1:5001/alive

# create the temp repo
set +x
curl -XPOST \
     -i \
     https://api.github.com/orgs/hail-ci-test/repos \
     -H "Authorization: token ${TOKEN}" \
     -d "{ \"name\" : \"${REPO_NAME}\" }"
set -x

../until-with-fuel 30 curl -fL https://github.com/hail-ci-test/${REPO_NAME}

# upload files to temp repo
# https://unix.stackexchange.com/questions/30091/fix-or-alternative-for-mktemp-in-os-x
REPO_DIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
cp test-repo/* ${REPO_DIR}
pushd ${REPO_DIR}
git init
git config user.email 'ci-automated-tests@broadinstitute.org'
git config user.name 'ci-automated-tests'
set +x
git remote add origin \
    https://${TOKEN}@github.com/hail-ci-test/${REPO_NAME}.git
set -x
git add *
git commit -m 'inital commit'
git push origin master:master
popd

# start CI system
python3 run_ci.py --debug & ci_pid=$!

sleep 10

# setup webhooks for temp repo
set +x
./setup-endpoints.sh hail-ci-test/${REPO_NAME} ${TOKEN} ${SELF_HOSTNAME}
set -x

PYTHONPATH=${PWD}:${PYTHONPATH} python3 -m pytest -vv --failed-first --maxfail=1 test/test-ci.py
