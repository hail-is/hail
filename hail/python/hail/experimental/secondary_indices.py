from hail.typecheck import typecheck, sized_tupleof, typecheck_method, sequenceof, nullable
from hail.expr.expressions.expression_typecheck import expr_any

import hail as hl

class SecondaryIndex:
    @typecheck_method(table_path=str, name=str, key_names=sequenceof(str))
    def __init__(self, table_path, name, key_names):
        self.table_path = table_path
        self.name = name
        self.key_names=key_names

    def query(self, key):
        return hl.query_table(self.table_path, key).map(lambda x: x.select(*self.key_names))


@typecheck(ht=hl.Table, index_field=str, destination=str)
def make_secondary_index(ht, index_field, destination):
    ht = ht.select(index_field).select_globals()
    orig_key = list(ht.key)
    ht = ht.key_by(index_field, *orig_key)
    ht.write(destination, overwrite=True)
    return SecondaryIndex(destination, index_field, orig_key)

@typecheck(base=str, queries=sized_tupleof(SecondaryIndex, expr_any), limit=nullable(int))
def read_secondary_indices(base,
                           *queries,
                           limit=None):
    keys_to_isect = [si.query(key) for si, key in queries]
    key = queries[0][0].key_names
    key_to_query = hl.keyed_intersection(*keys_to_isect, key=key)
    records = key_to_query.flatmap(lambda x: hl.query_table(base, x))
    if limit is not None:
        records = records[:limit]
    return records

