from google.cloud import storage
import re


def get_geometric_mean(prod_of_means, num_of_means):
    return prod_of_means ** (1.0 / num_of_means)


FILE_PATH_REGEX = re.compile(r'gs://((?P<bucket>[^/]+)/)(?P<path>.*)')


def parse_file_path(regex, name):
    match = regex.fullmatch(name)
    return match.groupdict()


def enumerate_list_index(list_of_lists):
    res_list = []
    i = 0
    for list in list_of_lists:
        temp = []
        temp = [i] * len(list)
        res_list.extend(temp)
        i += 1
    return res_list


class ReadGoogleStorage:
    def __init__(self, service_account_key_file=None):
        self.storage_client = storage.Client.from_service_account_json(service_account_key_file)

    def get_data_as_string(self, file_path):
        file_info = parse_file_path(FILE_PATH_REGEX, file_path)
        bucket = self.storage_client.get_bucket(file_info['bucket'])
        shorter_file_path = file_info['path']
        try:
            # get bucket data as blob
            blob = bucket.blob(shorter_file_path)
            # convert to string
            data = blob.download_as_string()
        except Exception:
            raise NameError()
        return data
