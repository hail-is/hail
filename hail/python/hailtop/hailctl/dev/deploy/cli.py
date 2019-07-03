def init_parser(parser):
    parser.add_argument(name='repo', type=str)
    parser.add_argument(name='branch', type=str)
    parser.add_argument(name='profile', type=str, choices=['batch_test'])

def main(args):
    pass