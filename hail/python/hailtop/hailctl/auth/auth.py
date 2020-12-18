from ..hailctl import hailctl


@hailctl.group(
    help="Manage Hail credentials.")
def auth():
    pass
