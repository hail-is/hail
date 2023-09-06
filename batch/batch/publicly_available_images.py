from typing import List


def publicly_available_images(docker_prefix: str) -> List[str]:
    return [
        f'{docker_prefix}/hailgenetics/{name}'
        for name in (
            'hail',
            'hailtop',
            'genetics',
            'python-dill',
            'vep-grch37-85',
            'vep-grch38-95',
        )
    ]
