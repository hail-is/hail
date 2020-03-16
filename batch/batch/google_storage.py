import re
import logging
import fnmatch
import glob
import google.api_core.exceptions
import google.oauth2.service_account
import google.cloud.storage
from hailtop.utils import blocking_to_async, retry_transient_errors


from .utils import FileSlice


logging.getLogger("google").setLevel(logging.WARNING)


wildcards = ('*', '?', '[', ']', '{', '}')


class GCS:
    @staticmethod
    def _parse_uri(uri):
        assert uri.startswith('gs://'), uri
        uri = uri.lstrip('gs://').split('/')
        bucket = uri[0]
        path = '/'.join(uri[1:])
        return bucket, path

    @staticmethod
    def _escape(path):
        new_path = []
        n = len(path)
        i = 0
        while i < n:
            if i <= n - 1 and path[i] == '\\' and path[i + 1] in wildcards:
                new_path.append('[')
                new_path.append(path[i + 1])
                new_path.append(']')
                i += 2
                continue

            new_path.append(path[i])
            i += 1
        return ''.join(new_path)

    @staticmethod
    def _contains_wildcard(c):
        i = 0
        n = len(c)
        while i < n:
            if i <= n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
                i += 2
                continue
            elif c[i] in wildcards:
                return True
            i += 1
        return False

    @staticmethod
    def _unescape_escaped_wildcards(c):
        new_c = []
        i = 0
        n = len(c)
        while i < n:
            if i <= n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
                new_c.append(c[i + 1])
                i += 2
                continue
            new_c.append(c[i])
            i += 1
        return ''.join(new_c)

    @staticmethod
    def _prefix_wout_wildcard(c):
        new_c = []
        i = 0
        n = len(c)
        while i < n:
            if i <= n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
                new_c.append(c[i + 1])
                i += 2
                continue
            elif c[i] in wildcards:
                return ''.join(new_c)
            new_c.append(c[i])
            i += 1
        return ''.join(new_c)

    def __init__(self, blocking_pool, *, project=None, credentials=None):
        self.blocking_pool = blocking_pool
        # project=None doesn't mean default, it means no project:
        # https://github.com/googleapis/google-cloud-python/blob/master/storage/google/cloud/storage/client.py#L86
        if project:
            self.gcs_client = google.cloud.storage.Client(
                project=project, credentials=credentials)
        else:
            self.gcs_client = google.cloud.storage.Client(
                credentials=credentials)
        self._wrapped_write_gs_file_from_string = self._wrap_network_call(GCS._write_gs_file_from_string)
        self._wrapped_write_gs_file_from_file = self._wrap_network_call(GCS._write_gs_file_from_file)
        self._wrapped_read_gs_file = self._wrap_network_call(GCS._read_gs_file)
        self._wrapped_read_binary_gs_file = self._wrap_network_call(GCS._read_binary_gs_file)
        self._wrapped_read_gs_file_to_file = self._wrap_network_call(GCS._read_gs_file_to_file)
        self._wrapped_delete_gs_file = self._wrap_network_call(GCS._delete_gs_file)
        self._wrapped_delete_gs_files = self._wrap_network_call(GCS._delete_gs_files)
        self._wrapped_copy_gs_file = self._wrap_network_call(GCS._copy_gs_file)
        self._wrapped_list_gs_files = self._wrap_network_call(GCS._list_gs_files)
        self._wrapped_compose_gs_file = self._wrap_network_call(GCS._compose_gs_file)
        self._wrapped_glob_gs_files = self._wrap_network_call(GCS._glob_gs_files)
        self._wrapped_get_blob = self._wrap_network_call(GCS._get_blob)

    async def write_gs_file_from_string(self, uri, string, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_write_gs_file_from_string,
                                            self, uri, string, *args, **kwargs)

    async def write_gs_file_from_file(self, uri, file_name, start, end, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_write_gs_file_from_file,
                                            self, uri, file_name, start, end, *args, **kwargs)

    async def read_gs_file(self, uri, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_read_gs_file,
                                            self, uri, *args, **kwargs)

    async def read_binary_gs_file(self, uri, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_read_binary_gs_file,
                                            self, uri, *args, **kwargs)

    async def read_gs_file_to_file(self, uri, file_name, offset, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_read_gs_file_to_file,
                                            self, uri, file_name, offset, *args, **kwargs)

    async def delete_gs_file(self, uri):
        return await retry_transient_errors(self._wrapped_delete_gs_file,
                                            self, uri)

    async def delete_gs_files(self, uri_prefix):
        return await retry_transient_errors(self._wrapped_delete_gs_files,
                                            self, uri_prefix)

    async def copy_gs_file(self, src, dest, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_copy_gs_file,
                                            self, src, dest, *args, **kwargs)

    async def compose_gs_file(self, sources, dest, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_compose_gs_file,
                                            self, sources, dest, *args, **kwargs)

    async def list_gs_files(self, uri, max_results=None):
        print('here')
        return await retry_transient_errors(self._wrapped_list_gs_files,
                                            self, uri, max_results=max_results)

    async def glob_gs_files(self, uri):
        return await retry_transient_errors(self._wrapped_glob_gs_files,
                                            self, uri)

    async def get_blob(self, uri):
        return await retry_transient_errors(self._wrapped_get_blob,
                                            self, uri)

    def _wrap_network_call(self, fun):
        async def wrapped(*args, **kwargs):
            return await blocking_to_async(self.blocking_pool,
                                           fun,
                                           *args,
                                           **kwargs)
        wrapped.__name__ = fun.__name__
        return wrapped

    def _write_gs_file_from_string(self, uri, string, *args, **kwargs):
        f = self._get_blob(uri)
        f.metadata = {'Cache-Control': 'no-cache'}
        f.upload_from_string(string, *args, **kwargs)

    def _write_gs_file_from_file(self, uri, file_name, start, end, *args, **kwargs):
        with FileSlice(file_name, start, end - start) as file:
            f = self._get_blob(uri)
            f.metadata = {'Cache-Control': 'no-cache'}
            f.upload_from_file(file, *args, **kwargs)

    def _read_gs_file(self, uri, *args, **kwargs):
        content = self._read_binary_gs_file(uri, *args, **kwargs)
        return content.decode('utf-8')

    def _read_binary_gs_file(self, uri, *args, **kwargs):
        f = self._get_blob(uri)
        f.metadata = {'Cache-Control': 'no-cache'}
        content = f.download_as_string(*args, **kwargs)
        return content

    def _read_gs_file_to_file(self, uri, file_name, offset, *args, **kwargs):
        with open(file_name, 'r+b') as file:
            file.seek(offset)
            f = self._get_blob(uri)
            f.metadata = {'Cache-Control': 'no-cache'}
            f.download_to_file(file, *args, **kwargs)

    def _delete_gs_files(self, uri_prefix):
        bucket, prefix = GCS._parse_uri(uri_prefix)
        bucket = self.gcs_client.bucket(bucket)
        for blob in bucket.list_blobs(prefix=prefix):
            try:
                blob.delete()
            except google.api_core.exceptions.NotFound:
                continue

    def _delete_gs_file(self, uri):
        f = self._get_blob(uri)
        try:
            f.delete()
        except google.api_core.exceptions.NotFound:
            return

    def _copy_gs_file(self, src, dest, *args, **kwargs):
        src_bucket, src_path = GCS._parse_uri(src)
        src_bucket = self.gcs_client.bucket(src_bucket)
        dest_bucket, dest_path = GCS._parse_uri(dest)
        dest_bucket = self.gcs_client.bucket(dest_bucket)
        src_f = src_bucket.blob(src_path)
        src_bucket.copy_blob(src_f, dest_bucket, new_name=dest_path, *args, **kwargs)

    def _list_gs_files(self, uri, max_results=None):
        bucket_name, prefix = GCS._parse_uri(uri)
        bucket = self.gcs_client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=prefix, max_results=max_results):
            yield blob

    def _glob_gs_files(self, uri):
        assert '**' not in uri

        bucket_name, path = GCS._parse_uri(uri)
        pattern = GCS._escape(path)
        # need a custom escape because escaped characters are not treated properly with glob.escape
        # and fnmatch doesn't work with escaped characters like \?

        components = path.rstrip('/').split('/')
        pattern_components = pattern.rstrip('/').split('/')

        # components = path.split('/')
        # pattern_components = pattern.split('/')

        def _glob(bucket, prefix, i):
            if i == len(components):
                blobs = {blob.name: blob
                         for blob in bucket.list_blobs(prefix=prefix, delimiter=None)}
                # print(f'prefix {prefix} i {i} blobs in _glob {blobs}')
                if not path.endswith('/') and path in blobs:
                    return [blobs[path]]
                return [blob for _, blob in blobs.items()
                        if fnmatch.fnmatchcase(GCS._unescape_escaped_wildcards(blob.name), pattern) or
                        fnmatch.fnmatchcase(GCS._unescape_escaped_wildcards(blob.name), pattern.rstrip('/')) or
                        fnmatch.fnmatchcase(GCS._unescape_escaped_wildcards(blob.name), pattern.rstrip('/') + '/*')]

            c = components[i]
            if i != len(components) - 1 and GCS._contains_wildcard(c):
                # print('not at end of path and contains wildcard')
                blobs = []
                if prefix:
                    prefix += '/'
                for page in bucket.list_blobs(prefix=prefix, delimiter='/').pages:
                    for new_prefix in page.prefixes:
                        # print(f'new_prefix {new_prefix}')
                        new_prefix = new_prefix.rstrip('/')
                        p = '/'.join(pattern_components[:i+1])
                        # print(f'fnmatch {new_prefix} {p}')
                        if fnmatch.fnmatchcase(GCS._unescape_escaped_wildcards(new_prefix), p):
                            blobs.extend(_glob(bucket, new_prefix, i + 1))
                return blobs

            c = GCS._prefix_wout_wildcard(c)
            c = GCS._unescape_escaped_wildcards(c)
            new_prefix = f'{prefix}/{c}' if prefix else c
            return _glob(bucket, new_prefix, i + 1)

        if '*' in bucket_name:
            bucket_prefix = GCS._prefix_wout_wildcard(bucket_name)
            buckets = [bucket for bucket in self.gcs_client.list_buckets(prefix=bucket_prefix)
                       if fnmatch.fnmatchcase(bucket.name, bucket_name)]
        else:
            buckets = [self.gcs_client.bucket(bucket_name)]

        blobs = [blob for bucket in buckets for blob in _glob(bucket, '', 0)]
        # print(f'result {blobs}')
        return blobs




        #
        # def _glob(bucket, prefix):
        #     print(bucket, prefix)
        #     print(prefix.split('/'), c.split('/'))
        #     if len(prefix.split('/')) == len(c.split('/')):
        #         blobs = {blob.name: blob
        #                  for blob in bucket.list_blobs(prefix=prefix.rstrip('/'), delimiter=None)}
        #         print(blobs)
        #         if not prefix.endswith('/') and prefix in blobs:
        #             return [blobs[prefix]]
        #         return [blob for _, blob in blobs.items()
        #                 if fnmatch.fnmatchcase(blob.name, pattern) or
        #                 fnmatch.fnmatchcase(blob.name, pattern.rstrip('/')) or
        #                 fnmatch.fnmatchcase(blob.name, pattern.rstrip('/') + '/*')]
        #
        #     blobs = []
        #     for page in bucket.list_blobs(prefix=prefix, delimiter='/').pages:
        #         for subdir in page.prefixes:
        #             subst_path = GCS._substitute_wildcards(subdir, c)
        #             print(f'subst_path {subst_path} pattern {c} recurse {fnmatch.fnmatchcase(subst_path, pattern)}')
        #             if fnmatch.fnmatchcase(subst_path, pattern):
        #                 blobs.extend(_glob(bucket, GCS._prefix_wout_wildcard(subst_path)))
        #
        #     print(blobs)
        #     return blobs
        #
        # if '*' in bucket_name:
        #     bucket_prefix = re.split(wildcard_pattern, bucket_name)[0]
        #     buckets = [bucket for bucket in self.gcs_client.list_buckets(prefix=bucket_prefix)
        #                if fnmatch.fnmatchcase(bucket.name, bucket_name)]
        # else:
        #     buckets = [self.gcs_client.bucket(bucket_name)]
        #
        # print(f'first prefix {GCS._prefix_wout_wildcard(c)}')
        # return [blob for bucket in buckets for blob in _glob(bucket, GCS._prefix_wout_wildcard(c))]

    # working for other test cases
    # def _glob_gs_files(self, uri):
    #     assert '**' not in uri
    #
    #     bucket_name, pattern = GCS._parse_uri(uri)
    #     pattern = glob.escape(pattern)
    #
    #     def _list_blobs(bucket, path):
    #         # prefix = re.split(wildcard_pattern, path)[0]
    #         prefix = GCS._prefix_wout_wildcard(GCS._escape_wildcard(path))
    #         has_wildcard = prefix != path
    #         print(f'prefix {prefix} has wildcard {has_wildcard}')
    #
    #         if not has_wildcard:
    #             blobs = {blob.name: blob
    #                      for blob in bucket.list_blobs(prefix=prefix.rstrip('/'), delimiter=None)}
    #             if not path.endswith('/') and path in blobs:
    #                 return [blobs[path]]
    #             return [blob for _, blob in blobs.items()
    #                     if fnmatch.fnmatchcase(blob.name, pattern) or
    #                     fnmatch.fnmatchcase(blob.name, pattern.rstrip('/')) or
    #                     fnmatch.fnmatchcase(blob.name, pattern.rstrip('/') + '/*')]
    #
    #         blobs = []
    #         for page in bucket.list_blobs(prefix=prefix, delimiter='/').pages:
    #             for subdir in page.prefixes:
    #                 # subdir = glob.escape(subdir)
    #                 # subdir = GCS._escape_wildcard(subdir)
    #                 subst_path = GCS._substitute_wildcards(subdir, pattern)
    #                 print(f'subst_path {subst_path} pattern {pattern} recurse {fnmatch.fnmatchcase(subst_path, pattern)}')
    #                 if fnmatch.fnmatchcase(subst_path, pattern):
    #                     blobs.extend(_list_blobs(bucket, subst_path))
    #
    #         print(blobs)
    #         return blobs
    #
    #     if '*' in bucket_name:
    #         bucket_prefix = re.split(wildcard_pattern, bucket_name)[0]
    #         buckets = [bucket for bucket in self.gcs_client.list_buckets(prefix=bucket_prefix)
    #                    if fnmatch.fnmatchcase(bucket.name, bucket_name)]
    #     else:
    #         buckets = [self.gcs_client.bucket(bucket_name)]
    #
    #     return [blob for bucket in buckets for blob in _list_blobs(bucket, pattern)]


    # def _glob_gs_files(self, uri):
    #     assert '**' not in uri
    #
    #     bucket_name, path = GCS._parse_uri(uri)
    #     # assert '**' not in bucket_name
    #     assert '?' not in bucket_name
    #
    #     if '*' in bucket_name:
    #         bucket_prefix = bucket_name.split('*')[0]
    #         buckets = [bucket for bucket in self.gcs_client.list_buckets(prefix=bucket_prefix)
    #                    if fnmatch.fnmatchcase(bucket.name, bucket_name)]
    #     else:
    #         buckets = [self.gcs_client.bucket(bucket_name)]
    #
    #     def substitute_wildcards(path):
    #         uri_components = uri.rstrip('/').split('/')
    #         path_components = path.rstrip('/').split('/')
    #         assert len(uri_components) >= len(path_components)
    #         new_path = path_components + uri_components[len(path_components):]
    #         return '/'.join(new_path)
    #
    #
    #     def _list(bucket, path):
    #         blobs = []
    #         prefix = re.split('\\*|\\[|\\?', path)[0]
    #         for blob in bucket.list_blobs(prefix=prefix, delimiter='/'):
    #             blob_path = 'gs://' + blob.bucket.name + '/' + blob.name
    #             if fnmatch.fnmatchcase(blob_path, uri):
    #                 blobs.append(blob)
    #             if fnmatch.fnmatchcase(blob_path, path):
    #                 blobs.extend(_list(bucket, blob_path))
    #         return blobs
    #
    #     blobs = []
    #     for bucket in buckets:
    #         blobs.extend(_list(bucket, path))
    #     return blobs


    # def _glob_gs_files(self, uri):
    #     assert '**' not in uri
    #
    #     bucket_name, path = GCS._parse_uri(uri)
    #     # assert '**' not in bucket_name
    #     assert '?' not in bucket_name
    #
    #     if '*' in bucket_name:
    #         bucket_prefix = bucket_name.split('*')[0]
    #         buckets = [bucket for bucket in self.gcs_client.list_buckets(prefix=bucket_prefix)
    #                    if fnmatch.fnmatchcase(bucket.name, bucket_name)]
    #     else:
    #         buckets = [self.gcs_client.bucket(bucket_name)]
    #
    #     def _list(bucket, path):
    #         blobs = []
    #         prefix = re.split('\\*|\\[|\\?', path)[0]
    #         for blob in bucket.list_blobs(prefix=prefix, delimiter='/'):
    #             blob_path = 'gs://' + blob.bucket.name + '/' + blob.name
    #             if fnmatch.fnmatchcase(blob_path, uri):
    #                 blobs.append(blob)
    #             if fnmatch.fnmatchcase(blob_path, path):
    #                 blobs.extend(_list(bucket, blob_path))
    #         return blobs
    #
    #     blobs = []
    #     for bucket in buckets:
    #         blobs.extend(_list(bucket, path))
    #     return blobs


    # def _glob_gs_files(self, uri):
    #     bucket_name, prefix = GCS._parse_uri(uri)
    #     assert '**' not in bucket_name
    #     assert '?' not in bucket_name
    #
    #     if '*' in bucket_name:
    #         bucket_prefix = bucket_name.split('*')[0]
    #         buckets = [bucket for bucket in self.gcs_client.list_buckets(prefix=bucket_prefix)
    #                    if fnmatch.fnmatchcase(bucket.name, bucket_name)]
    #     else:
    #         buckets = [self.gcs_client.bucket(bucket_name)]
    #
    #     if '*' in prefix:
    #         prefix = re.split('\\*|\\[|\\?', prefix)[0]
    #
    #     matches = {}
    #
    #     blobs = []
    #     for bucket in buckets:
    #         for blob in bucket.list_blobs(prefix=prefix):
    #             blob_path = 'gs://' + blob.bucket.name + '/' + blob.name
    #             dir_like_uri = uri.rstrip('/') + '/*'
    #             # need to address the problem where /foo/a/b and /foo/a
    #             # uri is /foo/a should only copy /foo/a and not /foo/a/b
    #             if (fnmatch.fnmatchcase(blob_path, uri) or
    #                     fnmatch.fnmatchcase(blob_path, dir_like_uri)):
    #                 blobs.append(blob)
    #     return blobs

    def _compose_gs_file(self, sources, dest, *args, **kwargs):
        assert sources
        sources = [self._get_blob(src) for src in sources]
        dest = self._get_blob(dest)
        dest.compose(sources, *args, **kwargs)

    def _get_blob(self, uri):
        bucket, path = GCS._parse_uri(uri)
        bucket = self.gcs_client.bucket(bucket)
        return bucket.blob(path)


    # ### WORKING ###
    # def _glob_gs_files(self, uri):
    #     print(uri)
    #     assert '**' not in uri
    #
    #     bucket_name, path = GCS._parse_uri(uri)
    #     pattern = GCS._escape(path)  # need this because escaped characters not treated properly with glob.escape
    #     print(f'pattern {pattern}')
    #     # and fnmatch doesn't work with escaped characters like \?
    #
    #     components = path.rstrip('/').split('/')
    #     pattern_components = pattern.rstrip('/').split('/')
    #     print(f'components == {components}')
    #
    #     def _contains_wildcard(c):
    #         i = 0
    #         n = len(c)
    #         while i < n:
    #             if i <= n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
    #                 i += 2
    #                 continue
    #             elif c[i] in wildcards:
    #                 return True
    #             i += 1
    #         return False
    #
    #     def _unescape_escaped_wildcards(c):
    #         new_c = []
    #         i = 0
    #         n = len(c)
    #         while i < n:
    #             if i <= n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
    #                 new_c.append(c[i + 1])
    #                 i += 2
    #                 continue
    #             new_c.append(c[i])
    #             i += 1
    #         return ''.join(new_c)
    #
    #     def _prefix_wout_wildcard(c):
    #         new_c = []
    #         i = 0
    #         n = len(c)
    #         while i < n:
    #             if i <= n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
    #                 new_c.append(c[i + 1])
    #                 i += 2
    #                 continue
    #             elif c[i] in wildcards:
    #                 return ''.join(new_c)
    #             new_c.append(c[i])
    #             i += 1
    #         return ''.join(new_c)
    #
    #     def _glob(bucket, prefix, i):
    #         print(f'prefix {prefix} i {i}')
    #         # assert not prefix.endswith('/')
    #
    #         if i == len(components):
    #             print('at end of path')
    #             # optimization get component before first wildcard character
    #             blobs = {blob.name: blob
    #                      for blob in bucket.list_blobs(prefix=prefix, delimiter=None)}
    #             if prefix and not prefix.endswith('/') and prefix in blobs:
    #                 return [blobs[prefix]]
    #             return [blob for _, blob in blobs.items()
    #                     if fnmatch.fnmatchcase(blob.name, pattern) or
    #                     fnmatch.fnmatchcase(blob.name, pattern.rstrip('/')) or
    #                     fnmatch.fnmatchcase(blob.name, pattern.rstrip('/') + '/*')]
    #
    #         component = components[i]
    #         print(f'component {component} i {i}')
    #         if i != len(components) - 1 and _contains_wildcard(component):
    #             print('contains wildcard')
    #             blobs = []
    #             # optimization get string before first wildcard character
    #             if prefix:
    #                 prefix += '/'
    #             print(f'making api call with prefix {prefix}')
    #             for page in bucket.list_blobs(prefix=prefix, delimiter='/').pages:
    #                 for new_prefix in page.prefixes:
    #                     new_prefix = new_prefix.rstrip('/')
    #                     p = '/'.join(pattern_components[:i+1])
    #                     print(new_prefix, p)
    #                     if fnmatch.fnmatchcase(new_prefix, p):
    #                         print('here')
    #                         blobs.extend(_glob(bucket, new_prefix, i + 1))
    #             return blobs
    #
    #         print(f'not last segment in path that matched wild card')
    #         print(f'original component {component}')
    #         component = _prefix_wout_wildcard(component)
    #         print(f'component prefix wout wildcard {component}')
    #         component = _unescape_escaped_wildcards(component)
    #         print(f'component without escaped wildcards {component}')
    #         new_prefix = f'{prefix}/{component}' if prefix else component
    #         print(f'new prefix {new_prefix}')
    #         return _glob(bucket, new_prefix, i + 1)