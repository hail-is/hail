import argparse
import time
import tempfile
from shlex import quote as shq

from hailtop.utils import secret_alnum_string, sync_check_shell_output


def check_if_db_clone_exists(*, dest_name, project):
    assert dest_name != "db-jcaq4"
    result, _ = sync_check_shell_output(f'''
gcloud sql instances list --filter name={shq(dest_name)} --project {shq(project)} --format="value(name)"
''',
                            echo=True)
    return result != b''


def check_if_vm_exists(*, vm_name, project):
    result, _ = sync_check_shell_output(f'''
gcloud compute instances list --filter "{shq(vm_name)}" --project {shq(project)} --format="value(name)"
''',
                            echo=True)
    return result != b''


def create_db_clone(*, source_name, dest_name, project):
    assert dest_name != "db-jcaq4"

    sync_check_shell_output(f'''
gcloud --project {shq(project)} sql instances clone {shq(source_name)} {shq(dest_name)}
''',
                            echo=True)


def setup_clone_user_ssl_certs(*, dest_name, project, db_username, password, tempdir):
    assert dest_name != "db-jcaq4"

    token = secret_alnum_string(5)

    db_username = db_username if db_username is not None else f'db-clone-root-{token}'
    password = secret_alnum_string(16) if password is None else password
    cert_name = dest_name + "-cert"

    server_ca_pem = f'{tempdir}/server-ca.pem'
    client_key_pem = f'{tempdir}/client-key.pem'
    client_cert_pem = f'{tempdir}/client-cert.pem'

    sync_check_shell_output(f'''
gcloud --project {shq(project)} sql users create {shq(db_username)} --instance={shq(dest_name)} --password="{shq(password)}"
gcloud beta --project {shq(project)} sql ssl server-ca-certs create --instance={shq(dest_name)}
gcloud beta --project {shq(project)} sql ssl server-ca-certs list --format="value(cert)" --instance={shq(dest_name)} > {server_ca_pem + '.'}
gcloud --project {shq(project)} sql ssl client-certs create {shq(cert_name)} {client_key_pem} --instance={shq(dest_name)}
gcloud --project {shq(project)} sql ssl client-certs describe {shq(cert_name)} --instance={shq(dest_name)} --format="value(cert)" > {client_cert_pem}
gcloud --project {shq(project)} sql instances describe {shq(dest_name)} --format="value(serverCaCert.cert)" > {server_ca_pem}
gcloud --project {shq(project)} sql instances describe {shq(dest_name)} --format="value(ipAddresses[].ipAddress)" > {tempdir}/dest_host
''',
                            echo=False)  # we don't want to echo the password into the terminal

    with open(f'{tempdir}/dest_host', 'r') as f:
        dest_host = f.read().rstrip()

    return (server_ca_pem, client_key_pem, client_cert_pem, db_username, password, dest_host)


def create_vm(*, vm_name, vm_cores, vm_disk_size, project, zone):
    assert vm_cores in (1, 2, 4, 8, 16, 32)
    assert 10 <= vm_disk_size <= 200

    sync_check_shell_output(f'''
gcloud -q compute instances create {shq(vm_name)} \
    --project {shq(project)}  \
    --zone={zone} \
    --machine-type=n1-standard-{vm_cores} \
    --network=default \
    --network-tier=PREMIUM \
    --metadata-from-file startup-script=setup-vm-with-clone-db-access.sh \
    --no-restart-on-failure \
    --maintenance-policy=MIGRATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --image='ubuntu-minimal-2004-focal-v20230303' \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size={vm_disk_size}GB \
    --boot-disk-type=pd-ssd
''',
                            echo=True)

    time.sleep(300)  # need to ensure startup script finishes running


def setup_db_certs_on_vm(*, tempdir, vm_name, server_ca_pem, client_key_pem, client_cert_pem, dest_name, dest_host, db_username, password, project, region, ssh_username):
    mysql_client_cnf = tempdir + '/mysql-config.cnf'
    mysql_client_json = tempdir + '/mysql-config.json'

    with open(mysql_client_cnf, 'w') as f:
        f.write(f'''[client]
host={dest_host}
user={db_username}
password="{password}"
database=batch
ssl-ca=/sql-config/server-ca.pem
ssl-cert=/sql-config/client-cert.pem
ssl-key=/sql-config/client-key.pem
ssl-mode=VERIFY_CA
''')

    with open(mysql_client_json, 'w') as f:
        f.write(f'''{{"host": "{dest_host}", "port": 3306, "user": "{db_username}", "password": "{password}", "instance": "{dest_name}", "connection_name": "{project}:{region}:{dest_name}", "db": "batch", "ssl-ca": "/sql-config/server-ca.pem", "ssl-cert": "/sql-config/client-cert.pem", "ssl-key": "/sql-config/client-key.pem", "ssl-mode": "VERIFY_CA"}}''')

    sync_check_shell_output(f'''
gcloud compute scp --project {shq(project)} {server_ca_pem} {ssh_username}@{vm_name}:/sql-config/
gcloud compute scp --project {shq(project)} {client_key_pem} {ssh_username}@{vm_name}:/sql-config/
gcloud compute scp --project {shq(project)} {client_cert_pem} {ssh_username}@{vm_name}:/sql-config/
gcloud compute scp --project {shq(project)} {mysql_client_cnf} {ssh_username}@{vm_name}:/sql-config/
gcloud compute scp --project {shq(project)} {mysql_client_json} {ssh_username}@{vm_name}:/sql-config/
''')


def main(args):
    assert args.dest_name != "db-jcaq4"

    vm_name = f'clone-db-vm-{args.dest_name}'

    db_exists = check_if_db_clone_exists(dest_name=args.dest_name, project=args.project)
    if not db_exists:
        print(f"creating clone db {args.dest_name} from source db {args.source_name}")
        create_db_clone(
            source_name=args.source_name,
            dest_name=args.dest_name,
            project=args.project,
        )
    else:
        print(f"clone db {args.dest_name} already exists. Doing nothing.")

    args.source_name = None

    vm_exists = check_if_vm_exists(vm_name=vm_name, project=args.project)
    if not vm_exists:
        print(f"creating vm {vm_name}")
        create_vm(
            vm_name=vm_name,
            vm_cores=args.vm_cores,
            vm_disk_size=args.vm_disk_size,
            project=args.project,
            zone=args.vm_zone,
        )
    else:
        print(f"vm {vm_name} already exists. Doing nothing.")

    with tempfile.TemporaryDirectory() as tempdir:
        server_ca_pem, client_key_pem, client_cert_pem, db_username, password, dest_host = setup_clone_user_ssl_certs(
            dest_name=args.dest_name,
            db_username=args.db_username,
            password=args.db_password,
            tempdir=tempdir,
            project=args.project,
        )

        setup_db_certs_on_vm(
            tempdir=tempdir,
            vm_name=vm_name,
            server_ca_pem=server_ca_pem,
            client_key_pem=client_key_pem,
            client_cert_pem=client_cert_pem,
            dest_name=args.dest_name,
            dest_host=dest_host,
            db_username=db_username,
            password=password,
            project=args.project,
            region=args.db_region,
            ssh_username=args.ssh_username
        )

    print(f'''created clone of {args.source_name}:
project: {args.project}
dest-db-name: {args.dest_name}
vm-name: {vm_name}
vm-cores: {args.vm_cores}
vm-disk-size: {args.vm_disk_size}
connect to db with "gcloud compute ssh {vm_name} --project {args.project}"
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--vm-zone', type=str, required=False, default='us-central1-a')
    parser.add_argument('--db-region', type=str, required=False, default='us-central1')
    parser.add_argument('--source-name', type=str, required=True)
    parser.add_argument('--dest-name', type=str, required=True)
    parser.add_argument('--vm-cores', type=int, default=4)
    parser.add_argument('--vm-disk-size', type=int, default=50)
    parser.add_argument('--ssh-username', type=str, required=True)
    parser.add_argument('--db-username', type=str, required=False)
    parser.add_argument('--db-password', type=str, required=False)

    args = parser.parse_args()

    main(args)
