import os
import traceback

from hailtop.batch_client.aioclient import BatchClient


def get_billing_project_prefix():
    return f'__testproject_{os.environ["HAIL_TOKEN"]}'


async def delete_all_test_billing_projects():
    billing_project_prefix = get_billing_project_prefix()
    bc = await BatchClient.create('', cloud_credentials_file=os.environ['HAIL_TEST_DEV_GSA_KEY_FILE'])
    try:
        for project in await bc.list_billing_projects():
            if project['billing_project'].startswith(billing_project_prefix):
                try:
                    print(f'deleting {project}')
                    await bc.delete_billing_project(project['billing_project'])
                except Exception:
                    print(f'exception deleting {project}; will continue')
                    traceback.print_exc()
    finally:
        await bc.close()
