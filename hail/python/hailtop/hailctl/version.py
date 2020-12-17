from . import version


def init_parser(subparsers):
    deploy_parser = subparsers.add_parser(
        'version',
        help='Print version',
        description='Print version')
    deploy_parser.set_defaults(module='hailctl version')


def main(args):
    import pkg_resources  # pylint: disable=import-outside-toplevel
    print(pkg_resources.resource_string(__name__, 'hail_version').decode().strip())
