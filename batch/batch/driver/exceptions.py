class RegionsNotSupportedError(Exception):
    def __init__(self, desired_regions, supported_regions):
        super().__init__(
            f'no regions given in {desired_regions} are supported. choose from a region in {supported_regions}'
        )


class LocalSSDNotSupportedError(Exception):
    def __init__(self, machine_family: str):
        super().__init__(f'the {machine_family} machine family supports no local SSDs')
