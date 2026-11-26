import os
import sys
from typing import List, Optional

from . import emr


def spark_submit_step_args(script_s3_uri: str, pass_through_args: List[str]) -> List[str]:
    return ['spark-submit', '--deploy-mode', 'client', script_s3_uri, *pass_through_args]


def submit(
    cluster_id: str,
    script: str,
    remote_tmpdir: str,
    region: Optional[str],
    pass_through_args: List[str],
    wait: bool = True,
) -> str:
    from hailtop.utils import secret_alnum_string  # pylint: disable=import-outside-toplevel

    resolved_region = emr.resolve_region(region)
    client = emr.emr_client(resolved_region)

    prefix = remote_tmpdir.rstrip('/') + '/' + secret_alnum_string(8)
    script_s3_uri = f'{prefix}/{os.path.basename(script)}'

    print(f'Uploading {script} to {script_s3_uri}')
    with open(script, 'rb') as f:
        emr.upload_to_s3(script_s3_uri, f.read())

    resp = client.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                'Name': f'hail-{os.path.basename(script)}',
                'ActionOnFailure': 'CONTINUE',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': spark_submit_step_args(script_s3_uri, pass_through_args),
                },
            }
        ],
    )
    step_id = resp['StepIds'][0]
    print(f'Submitted step {step_id} to cluster {cluster_id}.')

    if not wait:
        return step_id

    print('Waiting for step to complete ...')
    waiter = client.get_waiter('step_complete')
    try:
        # botocore's default caps the wait at 30 minutes (30s x 60), which is far
        # too short for a Hail job. Poll for up to a week instead.
        waiter.wait(ClusterId=cluster_id, StepId=step_id, WaiterConfig={'Delay': 30, 'MaxAttempts': 20160})
    except Exception as e:  # surface the failure to the user
        # The waiter's own message carries the state change reason AWS reported, so
        # print it whether or not the follow-up describe_step succeeds.
        print(f'Step {step_id} did not complete successfully: {e}', file=sys.stderr)
        try:
            desc = client.describe_step(ClusterId=cluster_id, StepId=step_id)
            state = desc['Step']['Status']['State']
            print(f'Step {step_id} state is {state}.', file=sys.stderr)
        except Exception:
            print(f'Could not describe step {step_id} to report its state.', file=sys.stderr)
        raise SystemExit(1) from e

    print(f'Step {step_id} completed.')
    return step_id
