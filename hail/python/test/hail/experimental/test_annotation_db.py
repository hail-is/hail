import unittest
import tempfile

import hail as hl
from ..helpers import startTestHailContext, stopTestHailContext


class AnnotationDBTests(unittest.TestCase):
    @classmethod
    def setupAnnotationDBTests(cls):
        startTestHailContext()
        t = hl.utils.range_table(10)
        t = t.annotate(locus=hl.locus('1', t.idx + 1))
        t = t.annotate(annotation=hl.str(t.idx))
        d = tempfile.TemporaryDirectory()
        fname = d.name + '/f.mt'
        t.write(fname)
        cls.temp_dir = d
        cls.db_json = {
            'unique_dataset': {'description': 'now with unique rows!',
                               'url': 'http://example.com',
                               'key_properties': ['unique'],
                               'versions': [{'url': fname, 'version': 'v1-GRCh37'}]},
            'nonunique_dataset': {'description': 'non-unique rows :(',
                                  'url': 'http://example.net',
                                  'key_properties': [],
                                  'versions': [{'url': fname, 'version': 'v1-GRCh37'}]}}

    def tearDownAnnotationDBTests():
        stopTestHailContext()
        AnnotationDBTests.temp_dir.cleanup()

    setUpClass = setupAnnotationDBTests
    tearDownClass = tearDownAnnotationDBTests

    def test_uniqueness(self):
        db = hl.experimental.DB(config=AnnotationDBTests.db_json)
        t = hl.utils.range_table(10)
        t = t.annotate(locus=hl.locus('1', t.idx + 1))
        t = db.annotate_rows_db(t, 'unique_dataset', 'nonunique_dataset')
        t.unique_dataset.dtype == hl.tstruct(annotation=hl.tstr)
        t.nonunique_dataset.dtype == hl.tstruct(annotation=hl.tarray(hl.tstr))
