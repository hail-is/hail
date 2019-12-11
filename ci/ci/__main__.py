from gear import configure_logging
# configure logging before importing anything else
configure_logging()


def main():
    from .ci import run
    run()


main()
