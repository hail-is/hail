from typing import Mapping, Any
from .base_client import BaseClient


def or_none(f, x):
    if x is None:
        return x
    return f(x)


def not_implemented(x):
    raise NotImplementedError(x)


class BigQueryException(Exception):
    pass


class ResultsParser:
    @staticmethod
    def _parse_struct(schema, data):
        d = {}
        for value, fd in zip(data['f'], schema['fields']):
            value = value['v']
            if fd['type'] in ('RECORD', 'STRUCT'):
                if fd['mode'] == 'REPEATED':
                    value = [ResultsParser._parse_struct(fd, v['v']) for v in value]
                else:
                    value = ResultsParser._parse_struct(fd, value)
            elif fd['type'] == 'STRING':
                value = or_none(str, value)
            elif fd['type'] in ('FLOAT', 'FLOAT64'):
                value = or_none(float, value)
            elif fd['type'] in ('INTEGER', 'INT64'):
                value = or_none(int, value)
            elif fd['type'] in ('BOOLEAN', 'BOOL'):
                value = or_none(bool, value)
            elif fd['type'] == 'TIMESTAMP':
                value = int(or_none(float, value))
            else:
                # DATE, TIME, DATETIME
                raise NotImplementedError((fd, value))

            d[fd['name']] = value
        return d

    def __init__(self, schema):
        self.schema = schema

    def parse_record(self, data):
        return ResultsParser._parse_struct(self.schema, data)


class PagedQueriesIterator:
    def __init__(self, client: 'BigQueryClient', query: str, request_kwargs: Mapping[str, Any]):
        self._client = client
        self._query = query
        self._request_kwargs = request_kwargs
        self._page = None
        self._parser = None
        self._job_id = None
        self._location = None
        self._row_index = None
        self._total_rows = None

    def __aiter__(self) -> 'PagedQueriesIterator':
        return self

    async def __anext__(self):
        if self._page is None:
            config = {
                'kind': 'bigquery#queryRequest',
                'useLegacySql': False,
                'query': self._query}

            self._page = await self._client.post(
                '/queries', json=config, **self._request_kwargs)
            self._row_index = 0
            self._total_rows = self._page['totalRows']
            self._parser = ResultsParser(self._page['schema'])
            self._job_id = self._page['jobReference']['jobId']
            self._location = self._page['jobReference']['location']

            if self._page.get('errors'):
                raise BigQueryException(self._page['errors'])

        # in case a response is empty but there are more pages
        while True:
            # an empty page has no rows
            if 'rows' in self._page and self._row_index < len(self._page['rows']):
                i = self._row_index
                self._row_index += 1
                return self._parser.parse_record(self._page['rows'][i])

            next_page_token = self._page.get('pageToken')
            if next_page_token is not None:
                query_parameters = {
                    'pageToken': next_page_token,
                    'location': self._location
                }
                self._page = await self._client.get_query_results(
                    self._job_id, query_parameters, **self._request_kwargs)
                self._row_index = 0
            else:
                raise StopAsyncIteration


class BigQueryClient(BaseClient):
    def __init__(self, project, **kwargs):
        super().__init__(f'https://bigquery.googleapis.com/bigquery/v2/projects/{project}', **kwargs)

    # docs:
    # https://cloud.google.com/bigquery/docs/reference
    # https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets
    # https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs
    # https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query

    async def query(self, query, **kwargs):
        return PagedQueriesIterator(self, query, kwargs)

    async def get_query_results(self, job_id, query_parameters, **kwargs):
        return await self.get(f'/queries/{job_id}', params=query_parameters, **kwargs)
