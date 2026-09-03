#!/bin/bash
# Manual EMR smoke test. Requires AWS credentials, an S3 bucket you can write to,
# and permission to create EMR clusters with the default roles.
#
# Usage: S3_SCRATCH=s3://my-bucket/hail-emr-test/ bash hail/scripts/test-emr.sh

set -ex

: "${S3_SCRATCH:?set S3_SCRATCH to an s3:// URI you can write to}"

cluster_name="hail-emr-smoke-$(date +%s)"
cluster_id=

cleanup() {
    exit_code=$?
    trap - EXIT
    set +e
    if [[ -z "$cluster_id" ]]; then
        for _ in {1..5}; do
            cluster_id=$(hailctl emr list | awk -F '\t' -v n="$cluster_name" '$3 == n {print $1; exit}')
            [[ -n "$cluster_id" ]] && break
            sleep 2
        done
    fi
    if [[ -n "$cluster_id" ]]; then
        hailctl emr stop "$cluster_id" || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT

start_output=$(hailctl emr start "$cluster_name" \
    --s3-scratch "$S3_SCRATCH" \
    --core-instance-count 1 \
    --run-job-flow-json '{"Instances": {"KeepJobFlowAliveWhenNoSteps": true}}')
echo "$start_output"
cluster_id=$(sed -n 's/^Started cluster \(j-[A-Z0-9]*\)\.$/\1/p' <<<"$start_output")
if [[ -z "$cluster_id" ]]; then
    echo "could not determine the EMR cluster id" >&2
    exit 1
fi

aws emr wait cluster-running --cluster-id "$cluster_id"

cat > /tmp/hail-emr-smoke.py <<'PY'
import os
assert os.environ.get('HAIL_CLOUD') == 'aws', f"HAIL_CLOUD={os.environ.get('HAIL_CLOUD')!r}, expected 'aws'"
import hail as hl
mt = hl.balding_nichols_model(3, 100, 100)
mt.rows().write('SCRATCH/out.ht', overwrite=True)
print('OK')
PY
sed -i "s#SCRATCH#${S3_SCRATCH%/}#" /tmp/hail-emr-smoke.py

hailctl emr submit "$cluster_id" /tmp/hail-emr-smoke.py --s3-scratch "$S3_SCRATCH"

hailctl emr stop "$cluster_id"
cluster_id=
trap - EXIT
echo "SMOKE TEST PASSED"
