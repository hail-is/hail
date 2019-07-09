from hailtop.gear import configure_logging
# configure logging before importing anything else
configure_logging()


def main():
    from .auth import run
    run()


main()
