from google.cloud import storage
import re


class ReadGoogleStorage:
    storage_client = storage.Client()

    FILE_PATH_REGEX = re.compile(r'gs://((?P<bucket>[^/]+)/)((?P<user>[^/]+)/)((?P<version>[^-]+)-)((?P<sha>[^-]+))(-(?P<tag>[^\.]+))?\.json')

    def remove_prefix(self, text, prefix):
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
            raise
        return json_data

        # Then we want a method that reads an arbitrary file from google storage and returns the data as a string.
        # You'll probably want to write a parser that figures out the bucket and then the rest of the path
        # from a file path.
