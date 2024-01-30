from typing import List

from .hail_genetics_images import HAIL_GENETICS_IMAGES
from ..utils.utils import ParsedDockerImageReference


NON_HAILGENETICS_MIRRORED_IMAGES = [
    'ubuntu:20.04',
    'ubuntu:22.04',
    'ubuntu:24.04',
]
DOCKER_HUB_IMAGES_MIRRORED_IN_HAIL_BATCH = HAIL_GENETICS_IMAGES + NON_HAILGENETICS_MIRRORED_IMAGES


def publicly_available_images(docker_prefix: str) -> List[str]:
    return [docker_prefix + '/' + image_name for image_name in DOCKER_HUB_IMAGES_MIRRORED_IN_HAIL_BATCH]


def is_docker_hub_image_mirrored_in_hail_batch(image_ref: ParsedDockerImageReference) -> bool:
    return image_ref.hosted_in('dockerhub') and image_ref.name() not in DOCKER_HUB_IMAGES_MIRRORED_IN_HAIL_BATCH
