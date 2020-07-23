from hailtop.hail_logging import configure_logging
# configure logging before importing anything else
configure_logging()


def main():
    from .query import run
    run()


main()
