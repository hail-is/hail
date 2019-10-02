import argparse
import subprocess as sp


def cleanup_db(args_):
    parser = argparse.ArgumentParser()

    parser.add_argument('name')

    parser.add_argument('--dry-run',
                        action='store_true')

    args = parser.parse_args(args_)

    script = f"""
gcloud sql instances delete {args.name}
kubectl -n db-benchmark delete secret {args.name}-sql-config
kubectl -n db-benchmark delete pod {args.name}-admin
"""

    if args.dry_run:
        print(script)
    else:
        try:
            sp.check_output(script, shell=True)
            print(f'deleted instance {args.name}')
        except sp.CalledProcessError as e:
            print(e)
            print(e.output)
            raise
