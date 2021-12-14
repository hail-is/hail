from .. import describe

init_parser = describe.init_parser


async def main(*args, **kwargs):
    await describe.main_after_parsing(*args, **kwargs)
    print('!!! `hailctl dataproc describe` is DEPRECATED. Please use `hailctl describe` instead. !!!')
