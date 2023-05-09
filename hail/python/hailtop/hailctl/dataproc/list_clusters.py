from . import gcloud


async def main(args, pass_through_args):  # pylint: disable=unused-argument
    gcloud.run(['dataproc', 'clusters', 'list', *pass_through_args])
