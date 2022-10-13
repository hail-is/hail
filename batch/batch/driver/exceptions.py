class RegionsNotSupportedError(Exception):
    def __init__(self, desired_regions, supported_regions):
        super().__init__(
            f'no regions given in {desired_regions} are supported. choose from a region in {supported_regions}'
        )
