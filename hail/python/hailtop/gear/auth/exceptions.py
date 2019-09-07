class NoTokenFileFoundError(Exception):
    def __init__(self, tokens_file):
        self.tokens_file = tokens_file
        self.message = (f'Cannot authenticate because no tokens file was found '
                        f'at {tokens_file}. Execute `hailctl auth login` to '
                        f'obtain tokens.')
        super().__init__(self.message)
