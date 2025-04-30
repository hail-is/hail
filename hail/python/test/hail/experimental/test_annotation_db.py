import pytest

import hail as hl


class TestAnnotationDB:
    @pytest.fixture(scope="class")
    def db_json(self):
        t = hl.utils.genomic_range_table(10)
        t = t.annotate(annotation=hl.str(t.locus.position - 1))
        tempdir_manager = hl.TemporaryDirectory()
        d = tempdir_manager.__enter__()
        fname = d + '/f.mt'
        t.write(fname)
        db_json = {
            'unique_dataset': {
                'description': 'now with unique rows!',
                'url': 'https://example.com',
                'annotation_db': {'key_properties': ['unique']},
                'versions': [
                    {
                        'url': {
                            "aws": {"eu": fname, "us": fname},
                            "gcp": {"europe-west1": fname, "us-central1": fname},
                        },
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
                        'url': {
                            "aws": {"eu": fname, "us": fname},
                            "gcp": {"europe-west1": fname, "us-central1": fname},
                        },
                        'version': 'v1',
                        'reference_genome': 'GRCh37',
                    }
                ],
            },
        }

        yield db_json

        tempdir_manager.__exit__(None, None, None)

    def test_uniqueness(self, db_json):
        db = hl.experimental.DB(region='us-central1', cloud='gcp', config=db_json)
        t = hl.utils.genomic_range_table(10)
        t = db.annotate_rows_db(t, 'unique_dataset', 'nonunique_dataset')
        assert t.unique_dataset.dtype == hl.dtype('struct{annotation: str}')
        assert t.nonunique_dataset.dtype == hl.dtype('array<struct{annotation: str}>')
