
import sys, json, re
import hail as hl
from io import StringIO
from subprocess import check_output

hail_data_root = 'gs://hail-datasets-hail-data'

paths = [x['path'] for x in hl.hadoop_ls(hail_data_root) if x['path'].endswith('.ht') or x['path'].endswith('.mt')]

def parse_struct_fields(struct, full_parent_name, fields_array):
    for child_name in struct.keys():
        full_child_name = full_parent_name + '.' + child_name
        child_field_type = re.sub(r'{.*}', '', str(struct[child_name].dtype))
        fields_array.append((full_child_name, child_field_type))
        if child_field_type == 'struct':
            parse_fields(struct[child_name], full_child_name, fields_array)

datasets = []
for path in paths:
    try:
        check_output(['gsutil', 'stat', f'{path}/_SUCCESS'])
    except:
        continue
    else:
        print(f'Loading {path}...')
        if path.endswith('.ht'):
            dataset = hl.read_table(path)
            dataset_type = 'Table'
            entry_fields = None
            col_fields = None
            col_key = None
            row_key = [x for x in dataset.key]
            row_fields = []
            for field_name in dataset.row_value:
                field_type = re.sub(r'{.*}', '', str(dataset[field_name].dtype))
                row_fields.append((field_name, field_type))
                if field_type == 'struct':
                    parse_struct_fields(dataset[field_name], field_name, row_fields)
        else:
            dataset = hl.read_matrix_table(path)
            dataset_type = 'MatrixTable'
            entry_fields = []
            for field_name in dataset.entry:
                field_type = re.sub(r'{.*}', '', str(dataset[field_name].dtype))
                entry_fields.append((field_name, field_type))
                if field_type == 'struct':
                    parse_struct_fields(dataset[field_name], field_name, entry_fields)
            col_key = [x for x in dataset.col_key]
            col_fields = []
            for field_name in dataset.col_value:
                field_type = re.sub(r'{.*}', '', str(dataset[field_name].dtype))
                col_fields.append((field_name, field_type))
                if field_type == 'struct':
                    parse_struct_fields(dataset[field_name], field_name, col_fields)
            row_key = [x for x in dataset.row_key]
            row_fields = []
            for field_name in dataset.row_value:
                field_type = re.sub(r'{.*}', '', str(dataset[field_name].dtype))
                row_fields.append((field_name, field_type))
                if field_type == 'struct':
                    parse_struct_fields(dataset[field_name], field_name, row_fields)

        global_fields = []
        for field_name in dataset.globals:
            field_type = re.sub(r'{.*}', '', str(dataset[field_name].dtype))
            global_fields.append((field_name, field_type))
            if field_type == 'struct':
                parse_struct_fields(dataset[field_name], field_name, global_fields)

        """
        stdout = sys.stdout
        schema = StringIO()
        sys.stdout = schema
        dataset.describe()
        sys.stdout = stdout
        schema = schema.getvalue()
        """

        metadata = dataset['metadata'].collect()[0]
        datasets.append({
            'path': path,
            'type': dataset_type,
            'name': metadata['name'],
            'version': metadata['version'],
            'reference_genome': metadata['reference_genome'],
            'n_rows': metadata['n_rows'],
            'n_cols': metadata['n_cols'] if 'n_cols' in metadata else None,
            'n_partitions': metadata['n_partitions'],
            #'schema': schema,
            'global_fields': global_fields,
            'row_fields': row_fields,
            'row_key': row_key, 
            'col_fields': col_fields,
            'col_key': col_key,
            'entry_fields': entry_fields
        })

with hl.hadoop_open(f'{hail_data_root}/datasets.json', 'w') as f:
        json.dump(datasets, f)
