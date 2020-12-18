from .hailctl import hailctl


@hailctl.command(
    help="Print version")
def version():
    import pkg_resources  # pylint: disable=import-outside-toplevel
    print(pkg_resources.resource_string(__name__, 'hail_version').decode().strip())
