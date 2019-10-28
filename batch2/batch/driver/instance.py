class Instance:
    @staticmethod
    def from_record(record):
        return Instance(
            record['id'], record['state'], record['name'], record['token'],
            record['cores_mcpu'], record['free_cores_mcpu'], record['ip_address'])

    def __init__(self, id, state, name, token, cores_mcpu, free_cores_mcpu, ip_address):
        self.id = id
        # pending, active, inactive, deleted
        self.state = state
        self.name = name
        self.token = token
        self.cores_mcpu = cores_mcpu
        self.free_cores_mcpu = free_cores_mcpu
        self.ip_address = ip_address

    def __str__(self):
        return f'instance {self.name}'
