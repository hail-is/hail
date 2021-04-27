import unittest

import hail as hl
from ..helpers import startTestHailContext, stopTestHailContext


class AnnotationDBTests(unittest.TestCase):
    @classmethod
    def setupAnnotationDBTests(cls):
        startTestHailContext()
        t = hl.utils.range_table(10)
        t = t.key_by(locus=hl.locus('1', t.idx + 1))
        t = t.annotate(annotation=hl.str(t.idx))
        cls.tempdir_manager = hl.TemporaryDirectory()
        d = cls.tempdir_manager.__enter__()
        fname = d + '/f.mt'
        t.write(fname)
        cls.db_json = {
            'unique_dataset': {
                'description': 'now with unique rows!',
                'url': 'https://example.com',
                'annotation_db': {'key_properties': ['unique']},
                'versions': [{
                    'url': {"aws": {"eu": fname, "us": fname},
                            "gcp": {"eu": fname, "us": fname}},
                    'version': 'v1',
                    'reference_genome': 'GRCh37'
                }]
            },
            'nonunique_dataset': {
                'description': 'non-unique rows :(',
                'url': 'https://example.net',
                'annotation_db': {'key_properties': []},
                'versions': [{
                    'url': {"aws": {"eu": fname, "us": fname},
                            "gcp": {"eu": fname, "us": fname}},
                    'version': 'v1',
                    'reference_genome': 'GRCh37'
                }]
            }
        }

    @classmethod
    def tearDownAnnotationDBTests(cls):
        stopTestHailContext()
        cls.tempdir_manager.__exit__(None, None, None)

    setUpClass = setupAnnotationDBTests
    tearDownClass = tearDownAnnotationDBTests

    def test_uniqueness(self):
        db = hl.experimental.DB(region='us', cloud='gcp', config=AnnotationDBTests.db_json)
        t = hl.utils.range_table(10)
        t = t.key_by(locus=hl.locus('1', t.idx + 1))
        t = db.annotate_rows_db(t, 'unique_dataset', 'nonunique_dataset')
        assert t.unique_dataset.dtype == hl.dtype('struct{idx: int32, annotation: str}')
        assert t.nonunique_dataset.dtype == hl.dtype('array<struct{idx: int32, annotation: str}>')
