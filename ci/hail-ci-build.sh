#!/bin/bash

set -ex

CI_LOG=${ARTIFACTS}/ci-test.log
CI_SUCCESS=${ARTIFACTS}/_CI_SUCCESS

SUCCESS='<span style="color:green;font-weight:bold">SUCCESS</span>'
FAILURE='<span style="color:red;font-weight:bold">FAILURE</span>'
SKIPPED='<span style="color:gray;font-weight:bold">SKIPPED</span>'
STOPPED='<span style="color:gray;font-weight:bold">STOPPED</span>'

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


on_exit() {
    trap "" INT TERM
    set +e
    CI_TEST_STATUS=$(get_status "${CI_SUCCESS}")

    cat <<EOF > ${ARTIFACTS}/ci.html
<body>
<table>	
<tbody>
<tr>
<td>${CI_TEST_STATUS}</td>
<td><a href='artifacts/ci-test.log'>CI test log</a></td>
</tr>
</tbody>
</table>
<body>
EOF
}

trap on_exit EXIT


make test-in-cluster 2&>1 > ${CI_LOG}

touch ${CI_SUCCESS}
