from ..hailctl import hailctl


@hailctl.group(
    help="Manage the Hail Batch service.")
def batch():
    pass
