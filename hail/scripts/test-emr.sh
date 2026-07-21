#!/bin/bash
# Manual EMR smoke test. Requires AWS credentials, an S3 bucket you can write to,
# and permission to create EMR clusters with the default roles.
#
# Usage: S3_SCRATCH=s3://my-bucket/hail-emr-test/ bash hail/scripts/test-emr.sh

set -ex

: "${S3_SCRATCH:?set S3_SCRATCH to an s3:// URI you can write to}"

cluster_name="hail-emr-smoke-$(date +%s)"

hailctl emr start "$cluster_name" \
    --s3-scratch "$S3_SCRATCH" \
    --core-instance-count 1 \
    --run-job-flow-json '{"Instances": {"KeepJobFlowAliveWhenNoSteps": true}}'

# start prints "Started cluster j-XXXX." — capture the id by listing.
cluster_id=$(hailctl emr list | awk -v n="$cluster_name" '$3 == n {print $1}' | head -n1)

aws emr wait cluster-running --cluster-id "$cluster_id"

cat > /tmp/hail-emr-smoke.py <<'PY'
import hail as hl
mt = hl.balding_nichols_model(3, 100, 100)
mt.rows().write('SCRATCH/out.ht', overwrite=True)
print('OK')
PY
sed -i "s#SCRATCH#${S3_SCRATCH%/}#" /tmp/hail-emr-smoke.py

hailctl emr submit "$cluster_id" /tmp/hail-emr-smoke.py --s3-scratch "$S3_SCRATCH"

hailctl emr stop "$cluster_id"
echo "SMOKE TEST PASSED"
