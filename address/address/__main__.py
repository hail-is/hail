from gear import configure_logging
# configure logging before importing anything else
configure_logging()


def main():
    from .address import run
    run()


main()
