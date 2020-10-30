import os
import datetime

HAIL_BENCHMARK_BUCKET_NAME = os.environ['HAIL_BENCHMARK_BUCKET_NAME']
START_POINT = os.environ.get('START POINT')
if START_POINT is None:
    now = datetime.datetime()
    start_point = now - datetime.timedelta(days=1)
