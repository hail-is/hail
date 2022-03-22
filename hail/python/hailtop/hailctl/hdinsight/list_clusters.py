import subprocess


def init_parser(parser):
    del parser


async def main(args, pass_through_args):
    del args
    subprocess.check_call(['az', 'hdinsight', 'list', *pass_through_args])
