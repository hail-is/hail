from hailtop.hail_logging import configure_logging
# configure logging before importing anything else
configure_logging()


def main():
    from .main import run
    run()


main()
