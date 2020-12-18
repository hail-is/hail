from ..hailctl import hailctl


@hailctl.group(
    help="Manage batches running on the batch service managed by the Hail team.")
def batch():
    pass
