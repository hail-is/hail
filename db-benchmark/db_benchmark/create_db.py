import argparse
import uuid
import subprocess as sp


def create_db(args_):
    parser = argparse.ArgumentParser()

    parser.add_argument('name', nargs='?', default=f'test-{uuid.uuid4().hex[:6]}')

    parser.add_argument('--tier', '-t',
                        type=str,
                        required=False,
                        default='db-n1-standard-1',
                        help='Machine type.')

    parser.add_argument('--zone', '-z',
                        type=str,
                        required=False,
                        default='us-central1-a',
                        help='Zone to create db in.')

    parser.add_argument('--database-version',
                        type=str,
                        default='MYSQL_5_7',
                        choices=['MYSQL_5_6', 'MYSQL_5_7'],
                        help='Database version to use.')

    parser.add_argument('--database-flags',
                        type=str,
                        required=False,
                        help='Database flags to use')

    parser.add_argument('--storage-type',
                        type=str,
                        required=False,
                        default='SSD',
                        choices=['SSD', 'HDD'])

    parser.add_argument('--storage-size',
                        type=str,
                        required=False,
                        default='10GB')

    parser.add_argument('--dry-run',
                        action='store_true')

    args = parser.parse_args(args_)

    user_name = f'user-{uuid.uuid4().hex[:8]}'
    password = uuid.uuid4().hex[:16]

    database_flags = f'--database-flags={args.database_flags}' \
        if args.database_flags else ''

    create_instance = f"""
gcloud -q beta sql instances create {args.name} \
  --tier={args.tier} \
  --no-assign-ip \
  --network=default \
  --zone={args.zone} \
  --storage-type={args.storage_type} \
  --storage-size={args.storage_size} \
  --database-version={args.database_version} \
  {database_flags}
"""

    script = f"""
set -e

{create_instance}

IP_ADDRESS=$(gcloud sql instances describe --format json {args.name} | \
  jq -r '.ipAddresses[] | select(.type=="PRIVATE") | .ipAddress')

gcloud -q sql databases create test -i {args.name}

gcloud -q sql users create {user_name} --instance={args.name} --password={password}

cat > sql-config.json <<EOF
{{
  "host": "$IP_ADDRESS",
  "port": 3306,
  "user": "{user_name}",
  "password": "{password}",
  "instance": "{args.name}",
  "connection_name": null,
  "db": "test"
}}
EOF

cat > sql-config.cnf <<EOF
[client]
host=$IP_ADDRESS
user={user_name}
password="{password}"
database=test
EOF

kubectl -n db-benchmark create secret generic {args.name}-sql-config \
  --from-file=sql-config.json \
  --from-file=sql-config.cnf

rm sql-config.json sql-config.cnf


python3 ../ci/jinja2_render.py '{{ \
"pod_name": "{args.name}-admin", \
"image": "gcr.io/hail-vdc/service-base:latest", \
"db_secret_name": "{args.name}-sql-config" \
}}' admin-pod.yaml admin-pod.yaml.out

kubectl -n db-benchmark apply -f admin-pod.yaml.out
rm admin-pod.yaml.out
"""

    if args.dry_run:
        print(script)
    else:
        try:
            sp.check_output(script, shell=True)
            print(f'created instance {args.name} with the following command: {create_instance}')
        except sp.CalledProcessError as e:
            print(e)
            print(e.output)
            raise
