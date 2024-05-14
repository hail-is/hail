import pytest

import hail as hl
from hail.backend.service_backend import ServiceBackend


class TestAnnotationDB:
    @pytest.fixture(scope="class")
    def db_json(init_hail):
        backend = hl.current_backend()
        if isinstance(backend, ServiceBackend):
            backend.batch_attributes = dict(name='setupAnnotationDBTests')
        t = hl.utils.genomic_range_table(10)
        t = t.annotate(annotation=hl.str(t.locus.position - 1))
        tempdir_manager = hl.TemporaryDirectory()
        d = tempdir_manager.__enter__()
        fname = d + '/f.mt'
        t.write(fname)
        if isinstance(backend, ServiceBackend):
            backend.batch_attributes = dict()
        db_json = {
            'unique_dataset': {
                'description': 'now with unique rows!',
                'url': 'https://example.com',
                'annotation_db': {'key_properties': ['unique']},
                'versions': [
                    {
                        'url': {"aws": {"eu": fname, "us": fname}, "gcp": {"eu": fname, "us": fname}},
                        'version': 'v1',
                        'reference_genome': 'GRCh37',
                    }
                ],
            },
            'nonunique_dataset': {
                'description': 'non-unique rows :(',
                'url': 'https://example.net',
                'annotation_db': {'key_properties': []},
                'versions': [
                    {
                        'url': {"aws": {"eu": fname, "us": fname}, "gcp": {"eu": fname, "us": fname}},
                        'version': 'v1',
                        'reference_genome': 'GRCh37',
                    }
                ],
            },
        }

        yield db_json

        tempdir_manager.__exit__(None, None, None)

    def test_uniqueness(self, db_json):
        db = hl.experimental.DB(region='us', cloud='gcp', config=db_json)
        t = hl.utils.genomic_range_table(10)
        t = db.annotate_rows_db(t, 'unique_dataset', 'nonunique_dataset')
        assert t.unique_dataset.dtype == hl.dtype('struct{annotation: str}')
        assert t.nonunique_dataset.dtype == hl.dtype('array<struct{annotation: str}>')
