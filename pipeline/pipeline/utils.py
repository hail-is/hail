import random, string


def get_sha(k):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=k))
