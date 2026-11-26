import os
import sys
from typing import List, Optional

import typer

from . import emr


def spark_submit_step_args(script_s3_uri: str, pass_through_args: List[str]) -> List[str]:
    return ['spark-submit', '--deploy-mode', 'client', script_s3_uri, *pass_through_args]


def resolve_script_uri(script: str, upload_uri: Optional[str]) -> str:
    """Return the S3 URI of the script, uploading a local file to `upload_uri` first."""
    if script.startswith('s3://'):
        return script
    if upload_uri is None:
        raise typer.BadParameter('EMR can only run scripts from S3. Pass an s3:// SCRIPT, or use --upload-to.')
    if not upload_uri.startswith('s3://'):
        raise typer.BadParameter('--upload-to must be an S3 URI such as s3://my-bucket/hail-scripts/.')
    script_s3_uri = f'{upload_uri.rstrip("/")}/{os.path.basename(script)}'
    print(f'Uploading {script} to {script_s3_uri}')
    with open(script, 'rb') as f:
        emr.upload_to_s3(script_s3_uri, f.read())
    return script_s3_uri


def submit(
    cluster_id: str,
    script: str,
    upload_uri: Optional[str],
    region: Optional[str],
    pass_through_args: List[str],
    wait: bool = True,
) -> str:
    client = emr.emr_client(region)
    script_s3_uri = resolve_script_uri(script, upload_uri)

    resp = client.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                'Name': f'hail-{os.path.basename(script_s3_uri)}',
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
        # botocore's default caps the wait at 30 minutes, which is too short for a Hail job.
        waiter.wait(ClusterId=cluster_id, StepId=step_id, WaiterConfig={'Delay': 30, 'MaxAttempts': 20160})
    except Exception as e:
        print(f'Step {step_id} did not complete successfully: {e}', file=sys.stderr)
        raise SystemExit(1) from e

    print(f'Step {step_id} completed.')
    return step_id
