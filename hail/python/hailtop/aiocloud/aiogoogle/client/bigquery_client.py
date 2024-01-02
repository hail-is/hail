from typing import Any, Dict, Mapping, Optional, Union
from ..credentials import GoogleCredentials
from ...common.credentials import AnonymousCloudCredentials
from ...common.session import Session
from ...common.base_client import CloudBaseClient


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
        def parse_field(name: str, value: Any, schema: Dict[str, Any]) -> Any:
            typ: str = schema['type']
            mode: str = schema['mode']

            if typ in ('RECORD', 'STRUCT'):
                if mode == 'REPEATED':
                    return [ResultsParser._parse_struct(schema, v['v']) for v in value]
                return ResultsParser._parse_struct(schema, value)
            if typ == 'STRING':
                return or_none(str, value)
            if typ in ('FLOAT', 'FLOAT64'):
                return or_none(float, value)
            if typ in ('INTEGER', 'INT64'):
                return or_none(int, value)
            if typ in ('BOOLEAN', 'BOOL'):
                return or_none(bool, value)
            if typ == 'TIMESTAMP':
                return int(or_none(float, value))
            # DATE, TIME, DATETIME
            raise NotImplementedError((name, value, typ, mode))
        return {
            field['name']: parse_field(field['name'], field['v'], field_schema)
            for field, field_schema in zip(data['f'], schema['fields'])
        }

    def __init__(self, schema):
        self.schema = schema

    def parse_record(self, data):
        return ResultsParser._parse_struct(self.schema, data)


class PagedQueriesIterator:
    def __init__(self, client: 'GoogleBigQueryClient', query: str, request_kwargs: Mapping[str, Any]):
        self._client = client
        self._query = query
        self._request_kwargs = request_kwargs
        self._page: Optional[Dict[str, Any]] = None
        self._parser: Optional[ResultsParser] = None
        self._job_id = None
        self._location = None
        self._row_index: Optional[int] = None
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
            assert self._row_index is not None
            assert self._page
            if 'rows' in self._page and self._row_index < len(self._page['rows']):
                i = self._row_index
                self._row_index += 1
                assert self._parser
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


class GoogleBigQueryClient(CloudBaseClient):
    def __init__(self,
                 project: str,
                 credentials: Optional[Union[GoogleCredentials, AnonymousCloudCredentials]] = None,
                 credentials_file: Optional[str] = None,
                 params: Optional[Mapping[str, str]] = None,
                 **kwargs):
        super().__init__(
            base_url=f'https://bigquery.googleapis.com/bigquery/v2/projects/{project}',
            session=Session(
                credentials=credentials or GoogleCredentials.from_file_or_default(credentials_file),
                params=params,
                **kwargs
            )
        )

    # docs:
    # https://cloud.google.com/bigquery/docs/reference
    # https://cloud.google.com/bigquery/docs/reference/rest/v2/datasets
    # https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs
    # https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query

    async def query(self, query, **kwargs):
        return PagedQueriesIterator(self, query, kwargs)

    async def get_query_results(self, job_id, query_parameters, **kwargs):
        return await self.get(f'/queries/{job_id}', params=query_parameters, **kwargs)
