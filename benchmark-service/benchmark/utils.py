from google.cloud import storage
import re


def get_geometric_mean(prod_of_means, num_of_means):
    return prod_of_means ** (1.0 / num_of_means)


class ReadGoogleStorage:
    storage_client = storage.Client()

    FILE_PATH_REGEX = re.compile(r'gs://((?P<bucket>[^/]+)/)((?P<user>[^/]+)/)((?P<version>[^-]+)-)((?P<sha>[^-]+))(-(?P<tag>[^\.]+))?\.json')

    @staticmethod
    def remove_prefix(text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text

    def parse_file_path(self, name):
        match = self.FILE_PATH_REGEX.fullmatch(name)
        return match.groupdict()

    def get_data_as_string(self, file_path):
        file_info = self.parse_file_path(file_path)
        bucket = self.storage_client.get_bucket(file_info['bucket'])
        shorter_file_path = self.remove_prefix(file_path, 'gs://' + file_info['bucket'] + '/')

        try:
            # get bucket data as blob
            blob = bucket.blob(shorter_file_path)
            # convert to string
            json_data = blob.download_as_string()
        except Exception:
            raise NameError()
        return json_data
