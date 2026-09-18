#!/bin/bash
# Manual EMR Spark 4 smoke test.
# Required environment:
#   S3_SCRATCH, ARTIFACT_MANIFEST, SUBNET_ID,
#   SERVICE_ACCESS_SECURITY_GROUP, PRIMARY_SECURITY_GROUP, CORE_SECURITY_GROUP,
#   SERVICE_ROLE, INSTANCE_PROFILE
# Optional: AWS_REGION (default us-east-1)

set -ex

: "${S3_SCRATCH:?set S3_SCRATCH to an s3:// URI you can write to}"
: "${ARTIFACT_MANIFEST:?set ARTIFACT_MANIFEST to the local artifact manifest path}"
: "${SUBNET_ID:?set SUBNET_ID to a private NAT-backed subnet}"
: "${SERVICE_ACCESS_SECURITY_GROUP:?set SERVICE_ACCESS_SECURITY_GROUP to the EMR endpoint security group}"
: "${PRIMARY_SECURITY_GROUP:?set PRIMARY_SECURITY_GROUP to the private primary-node security group}"
: "${CORE_SECURITY_GROUP:?set CORE_SECURITY_GROUP to the private core/task-node security group}"
: "${SERVICE_ROLE:?set SERVICE_ROLE to the custom EMR service role name}"
: "${INSTANCE_PROFILE:?set INSTANCE_PROFILE to the custom EC2 instance profile name}"
REGION=${AWS_REGION:-us-east-1}

cluster_name="hail-emr-spark4-smoke-$(date +%s)"
cluster_id=

cleanup() {
    exit_code=$?
    trap - EXIT
    set +e
    if [[ -z "$cluster_id" ]]; then
        for _ in {1..5}; do
            cluster_id=$(hailctl emr list --region "$REGION" | awk -F '\t' -v n="$cluster_name" '$3 == n {print $1; exit}')
            [[ -n "$cluster_id" ]] && break
            sleep 2
        done
    fi
    if [[ -n "$cluster_id" ]]; then
        hailctl emr stop "$cluster_id" --region "$REGION" || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT

start_output=$(hailctl emr start "$cluster_name" \
    --artifact-manifest "$ARTIFACT_MANIFEST" \
    --s3-scratch "$S3_SCRATCH" \
    --region "$REGION" \
    --subnet-id "$SUBNET_ID" \
    --service-access-security-group "$SERVICE_ACCESS_SECURITY_GROUP" \
    --primary-security-group "$PRIMARY_SECURITY_GROUP" \
    --core-security-group "$CORE_SECURITY_GROUP" \
    --no-use-default-roles \
    --service-role "$SERVICE_ROLE" \
    --instance-profile "$INSTANCE_PROFILE" \
    --core-instance-count 1 \
    --idle-timeout 900)
echo "$start_output"
cluster_id=$(sed -n 's/^Started cluster \(j-[A-Z0-9]*\)\.$/\1/p' <<<"$start_output")
if [[ -z "$cluster_id" ]]; then
    echo "could not determine the EMR cluster id" >&2
    exit 1
fi

aws emr wait cluster-running --region "$REGION" --cluster-id "$cluster_id"

cat > /tmp/hail-emr-smoke.py <<'PY'
import os
import sys

assert os.environ.get('HAIL_CLOUD') == 'aws', os.environ.get('HAIL_CLOUD')
assert sys.version_info[:2] == (3, 12), sys.version
import hail as hl

mt = hl.balding_nichols_model(3, 100, 100)
mt.rows().write('SCRATCH/out.ht', overwrite=True)
print('OK')
PY
sed -i "s#SCRATCH#${S3_SCRATCH%/}#" /tmp/hail-emr-smoke.py

hailctl emr submit "$cluster_id" /tmp/hail-emr-smoke.py --region "$REGION" --s3-scratch "$S3_SCRATCH"

hailctl emr stop "$cluster_id" --region "$REGION"
cluster_id=
trap - EXIT
echo "SMOKE TEST PASSED"
