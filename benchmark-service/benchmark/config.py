import os

BENCHMARK_TEST_BUCKET_NAME = os.environ.get('HAIL_TEST_BENCHMARK_BUCKET_NAME') is not None
START_POINT = os.environ.get('START POINT') is not None
