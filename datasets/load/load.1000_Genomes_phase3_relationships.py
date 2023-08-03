
import hail as hl

ht_relationships = hl.import_table(
    'gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_sample_relationships.tsv.bgz')

ht_relationships = ht_relationships.rename({'Family ID': 'family_id',
                                            'Individual ID': 's',
                                            'Paternal ID': 'paternal_id',
                                            'Maternal ID': 'maternal_id',
                                            'Relationship': 'relationship_role',
                                            'Siblings': 'sibling_ids',
                                            'Second Order': 'second_order_relationship_ids',
                                            'Third Order': 'third_order_relationship_ids',
                                            'Children': 'children_ids'})
ht_relationships = ht_relationships.annotate(paternal_id=hl.or_missing(ht_relationships['paternal_id'] != '0',
                                                                       ht_relationships['paternal_id']),
                                             maternal_id=hl.or_missing(ht_relationships['maternal_id'] != '0',
                                                                       ht_relationships['maternal_id']),
                                             relationship_role=hl.if_else(ht_relationships['relationship_role'] == 'unrel',
                                                                          'unrelated',
                                                                          ht_relationships['relationship_role']),
                                             sibling_ids=hl.or_missing(ht_relationships['sibling_ids'] == '0',
                                                                       hl.map(lambda x: x.strip(), ht_relationships['sibling_ids'].split(','))),
                                             children_ids=hl.or_missing(ht_relationships['children_ids'] == '0',
                                                                        hl.map(lambda x: x.strip(), ht_relationships['children_ids'].split(','))),
                                             second_order_relationship_ids=hl.or_missing(ht_relationships['second_order_relationship_ids'] == '0',
                                                                                         hl.map(lambda x: x.strip(), ht_relationships['second_order_relationship_ids'].split(','))),
                                             third_order_relationship_ids=hl.or_missing(ht_relationships['third_order_relationship_ids'] == '0',
                                                                                        hl.map(lambda x: x.strip(), ht_relationships['third_order_relationship_ids'].split(','))))
ht_relationships = ht_relationships.key_by('s')
ht_relationships = ht_relationships.select('family_id',
                                           'relationship_role',
                                           'maternal_id',
                                           'paternal_id',
                                           'children_ids',
                                           'sibling_ids',
                                           'second_order_relationship_ids',
                                           'third_order_relationship_ids')

n_rows = ht_relationships.count()
n_partitions = ht_relationships.n_partitions()

ht_relationships = ht_relationships.annotate_globals(
    metadata=hl.struct(
        name='1000_Genomes_phase3_sample_relationships',
        n_rows=n_rows,
        n_partitions=n_partitions))

ht_relationships.write('gs://hail-datasets-hail-data/1000_Genomes_phase3_sample_relationships.ht', overwrite=True)
