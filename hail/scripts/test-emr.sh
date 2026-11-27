#!/bin/bash
# Manual EMR Spark 4 smoke test.
# Required environment: WHEEL (s3:// URI of a Hail wheel), S3_SCRIPTS (s3:// prefix you can write to)
# Optional: SUBNET_ID, SERVICE_ACCESS_SECURITY_GROUP, AWS_REGION

set -ex

: "${WHEEL:?set WHEEL to the s3:// URI of a Hail wheel built for Spark 4.1}"
: "${S3_SCRIPTS:?set S3_SCRIPTS to an s3:// prefix you can write to}"

args=(--wheel "$WHEEL" --num-workers 1 --idle-timeout 900)
region_args=()
[[ -n "${AWS_REGION:-}" ]] && region_args=(--region "$AWS_REGION")
[[ -n "${SUBNET_ID:-}" ]] && args+=(--subnet-id "$SUBNET_ID")
[[ -n "${SERVICE_ACCESS_SECURITY_GROUP:-}" ]] && args+=(--service-access-security-group "$SERVICE_ACCESS_SECURITY_GROUP")

cluster_name="hail-emr-spark4-smoke-$(date +%s)"
cluster_id=

cleanup() {
    exit_code=$?
    trap - EXIT
    set +e
    if [[ -z "$cluster_id" ]]; then
        for _ in {1..5}; do
            cluster_id=$(hailctl emr list "${region_args[@]}" | awk -F '\t' -v n="$cluster_name" '$3 == n {print $1; exit}')
            [[ -n "$cluster_id" ]] && break
            sleep 2
        done
    fi
    if [[ -n "$cluster_id" ]]; then
        hailctl emr stop "$cluster_id" "${region_args[@]}" || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT

start_output=$(hailctl emr start "$cluster_name" "${args[@]}" "${region_args[@]}")
echo "$start_output"
cluster_id=$(sed -n 's/^Started cluster \(j-[A-Z0-9]*\)\.$/\1/p' <<<"$start_output")
if [[ -z "$cluster_id" ]]; then
    echo "could not determine the EMR cluster id" >&2
    exit 1
fi

aws emr wait cluster-running "${region_args[@]}" --cluster-id "$cluster_id"

smoke=$(mktemp --suffix=.py)
cat > "$smoke" <<PY
import os
import sys

assert os.environ.get('HAIL_CLOUD') == 'aws', os.environ.get('HAIL_CLOUD')
assert sys.version_info[:2] == (3, 12), sys.version
import hail as hl

mt = hl.balding_nichols_model(3, 100, 100)
mt.rows().write('${S3_SCRIPTS%/}/smoke-out.ht', overwrite=True)
print('OK')
PY

hailctl emr submit "$cluster_id" "$smoke" "${region_args[@]}" --upload-to "$S3_SCRIPTS"

hailctl emr stop "$cluster_id" "${region_args[@]}"
cluster_id=
trap - EXIT
echo "SMOKE TEST PASSED"
