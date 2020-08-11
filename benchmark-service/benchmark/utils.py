from google.cloud import storage
import re


def get_geometric_mean(prod_of_means, num_of_means):
    return prod_of_means ** (1.0 / num_of_means)

# FILE_PATH_REGEX = re.compile(r'gs://((?P<bucket>[^/]+)/)((?P<user>[^/]+)/)((?P<version>[^-]+)-)((?P<sha>[^-]+))(-(?P<tag>[^\.]+))?\.json')


FILE_PATH_REGEX = re.compile(r'gs://((?P<bucket>[^/]+)/)(?P<path>*)')


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
    def __init__(self):
        self.storage_client = storage.Client()

    # @staticmethod
    # def remove_prefix(text, prefix):
    #     if text.startswith(prefix):
    #         return text[len(prefix):]
    #     return text

    def get_data_as_string(self, file_path):
        file_info = parse_file_path(FILE_PATH_REGEX, file_path)
        bucket = self.storage_client.get_bucket(file_info['bucket'])
        shorter_file_path = file_info['path']
            # ReadGoogleStorage.remove_prefix(file_path, 'gs://' + file_info['bucket'] + '/')

        try:
            # get bucket data as blob
            blob = bucket.blob(shorter_file_path)
            # convert to string
            data = blob.download_as_string()
        except Exception:
            raise NameError()
        return data
