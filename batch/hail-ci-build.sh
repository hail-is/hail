#!/bin/bash
set -ex

source activate hail-batch

BATCH_LOG=${ARTIFACTS}/batch-test.log
BATCH_SUCCESS=${ARTIFACTS}/_BATCH_SUCCESS

SUCCESS='<span style="color:green;font-weight:bold">SUCCESS</span>'
FAILURE='<span style="color:red;font-weight:bold">FAILURE</span>'
SKIPPED='<span style="color:gray;font-weight:bold">SKIPPED</span>'

get_status() {
    FILE_LOC=$1
    DEPENDENCY=$2
    if [ -n "${DEPENDENCY}" ] && [ "${DEPENDENCY}" != "${SUCCESS}" ]; then
        echo ${SKIPPED};
    elif [ -e ${FILE_LOC} ]; then
        echo ${SUCCESS};
    else echo ${FAILURE};
    fi
}

cleanup() {
    set - INT TERM
    set +e
    kill $(cat batch.pid)
    rm -rf batch.pid

    BATCH_TEST_STATUS=$(get_status "${BATCH_SUCCESS}")

    cat <<EOF > ${ARTIFACTS}/batch.html
<body>
<table>	
<tbody>
<tr>
<td>${BATCH_TEST_STATUS}</td>
<td><a href='artifacts/batch-test.log'>Batch test log</a></td>
</tr>
</tbody>
</table>
<body>
EOF
}
trap cleanup EXIT

trap "exit 24" INT TERM

# run the server in the background with in-cluster config
python batch/server.py & echo $! > batch.pid

sleep 5

POD_IP='127.0.0.1' BATCH_URL='http://127.0.0.1:5000' python -m unittest test/test_batch.py 2&>1 > ${BATCH_LOG}

touch ${BATCH_SUCCESS}