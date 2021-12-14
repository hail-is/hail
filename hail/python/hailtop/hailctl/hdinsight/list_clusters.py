import subprocess


def init_parser(parser):
    pass


async def main(args, pass_through_args):
    subprocess.check_call(['az', 'hdinsight', 'list'])
