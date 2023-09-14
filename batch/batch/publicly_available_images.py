from typing import List

from hailtop.batch.hail_genetics_images import HAIL_GENETICS_IMAGES


def publicly_available_images(docker_prefix: str) -> List[str]:
    return [docker_prefix + '/' + image_name for image_name in HAIL_GENETICS_IMAGES]
