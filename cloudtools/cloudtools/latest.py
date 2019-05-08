from .utils import latest_sha, load_config

def init_parser(parser):
    parser.add_argument('version', type=str, choices=['0.1', '0.2'],
                        help='Hail version to use (default: %(default)s).')
    parser.add_argument('spark', type=str,
                        help='Spark version used to build Hail (default: 2.2.0 for 0.2 and 2.0.2 for 0.1)')
    parser.add_argument('--sha', action='store_true', help="Print the newest deployed SHA.")
    parser.add_argument('--jar', action='store_true', help="Print the location of the newest deployed jar.")
    parser.add_argument('--zip', action='store_true', help="Print the location of the newest deployed zip.")

def main(args):
    sha = latest_sha(args.version, args.spark)
    if args.sha:
        print(sha)
    if args.jar or args.zip:
        config = load_config(sha, args.version)
        config.configure(sha, args.spark)
        if args.jar:
            print(config.jar())
        if args.zip:
            print(config.zip())
