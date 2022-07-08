import hail as hl
avro_file = 'gs://hail-common/test-resources/weather.avro'
from avro.datafile import DataFileReader
from avro.io import DatumReader

fs = hl.current_backend().fs

with DataFileReader(fs.open(avro_file, 'rb'), DatumReader()) as avro:
    expected = list(avro)

data = hl.import_avro([avro_file]).collect()
data = [dict(**s) for s in data]
assert expected == data
