def init_logging():
    import logging

    logging.basicConfig(format="%(asctime)-15s: %(levelname)s: %(message)s", level=logging.INFO)
