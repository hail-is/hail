from .batch_configuration import PROJECT

PUBLIC_GCR_IMAGES = (
    f'gcr.io/{PROJECT}/{name}'
    for name in ('query',))
