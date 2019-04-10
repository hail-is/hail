#!/bin/bash
# do not execute this file, use the Makefile

set -ex

cleanup() {
    set +e
    trap "" INT TERM
    [[ -z $proxy_pid ]] || kill -9 $proxy_pid
}
trap cleanup EXIT
trap "exit 24" INT TERM

if [[ -z $IN_HAIL_CI ]]; then
    connection_name=$(jq -r '.connection_name' $CLOUD_SQL_CONFIG_PATH)
    host=$(jq -r '.host' $CLOUD_SQL_CONFIG_PATH)
    port=$(jq -r '.port' $CLOUD_SQL_CONFIG_PATH)
    ./cloud_sql_proxy -instances=$connection_name=tcp:$port &
    proxy_pid=$!
    ../until-with-fuel 30 curl -fL $host:$port
fi

for table in jobs jobs-parents batch batch-jobs; do
    python3 -c "from batch.database import Database; db = Database.create_synchronous(\"$CLOUD_SQL_CONFIG_PATH\"); db.drop_table_sync(\"$table\"); assert not db.has_table_sync(\"$table\")"
done
