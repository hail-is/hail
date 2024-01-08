import hail as hl
from hail import Table
from hail.typecheck import nullable, oneof, sequenceof, typecheck
from hail.utils import new_temp_file, wrap_to_list


@typecheck(ht=Table, key=str, value=str, fields=str)
def gather(ht, key, value, *fields) -> Table:
    """Collapse fields into key-value pairs.

    :func:`.gather` mimics the functionality of the `gather()` function found in R's
    ``tidyr`` package. This is a way to turn "wide" format data into "long"
    format data.

    Parameters
    ----------
    ht : :class:`.Table`
        A Hail table.
    key : :class:`str`
        The name of the key field in the gathered table.
    value : :class:`str`
        The name of the value field in the gathered table.
    fields : variable-length args of obj:`str`
        Names of fields to gather in ``ht``.

    Returns
    -------
    :class:`.Table`
        Table with original ``fields`` gathered into ``key`` and ``value`` fields."""

    ht = ht.annotate(_col_val=hl.array([hl.struct(field_name=field, value=ht[field]) for field in fields]))
    ht = ht.drop(*fields)
    ht = ht.explode(ht['_col_val'])
    ht = ht.annotate(**{key: ht['_col_val'][0], value: ht['_col_val'][1]})
    ht = ht.drop('_col_val')

    ht_tmp = new_temp_file()
    ht.write(ht_tmp)

    return hl.read_table(ht_tmp)


@typecheck(ht=Table, field=str, value=str, key=nullable(oneof(str, sequenceof(str))))
def spread(ht, field, value, key=None) -> Table:
    """Spread a key-value pair of fields across multiple fields.

    :func:`.spread` mimics the functionality of the `spread()` function in R's
    `tidyr` package. This is a way to turn "long" format data into "wide"
    format data.

    Given a ``field``, :func:`.spread` will create a new table by grouping
    ``ht`` by its row key and, optionally, any additional fields passed to the
    ``key`` argument.

    After collapsing ``ht`` by these keys, :func:`.spread` creates a new row field
    for each unique value of ``field``, where the row field values are given by the
    corresponding ``value`` in the original ``ht``.


    Parameters
    ----------
    ht : :class:`.Table`
        A Hail table.
    field : :class:`str`
        The name of the factor field in `ht`.
    value : :class:`str`
        The name of the value field in `ht`.
    key : optional, obj:`str` or list of :class:`str`
        The name of any fields to group by, in addition to the
        row key fields of ``ht``.

    Returns
    -------
    :class:`.Table`
        Table with original ``key`` and ``value`` fields spread across multiple columns."""

    if key is None:
        key = list(ht.key)
    else:
        key = wrap_to_list(key)
        key = list(ht.key) + key

    field_vals = list(ht.aggregate(hl.agg.collect_as_set(ht[field])))
    ht = ht.group_by(*key).aggregate(
        **{rv: hl.agg.take(ht[rv], 1)[0] for rv in ht.row_value if rv not in set([*key, field, value])},
        **{
            fv: hl.agg.filter(
                ht[field] == fv,
                hl.rbind(hl.agg.take(ht[value], 1), lambda take: hl.if_else(hl.len(take) > 0, take[0], 'NA')),
            )
            for fv in field_vals
        }
    )

    ht_tmp = new_temp_file()
    ht.write(ht_tmp)

    return ht


@typecheck(ht=Table, field=str, into=sequenceof(str), delim=oneof(str, int))
def separate(ht, field, into, delim) -> Table:
    """Separate a field into multiple fields by splitting on a delimiter
    character or position.

    :func:`.separate` mimics the functionality of the `separate()` function in R's
    ``tidyr`` package.

    This function will create a new table where ``field`` has been split into
    multiple new fields, whose names are given by ``into``.

    If ``delim`` is a ``str`` (including regular expression strings), ``field``
    will be separated into columns by that string. In this case, the length
    of ``into`` must match the number of resulting fields.

    If ``delim`` is an ``int``, ``field`` will be separated into two row fields,
    where the first field contains the first ``delim`` characters of ``field``
    and the second field contains the remaining characters.

    Parameters
    ----------
    ht : :class:`.Table`
        A Hail table.
    field : :class:`str`
        The name of the field to separate in ``ht``.
    into : list of :class:`str`
        The names of the fields to create by separating ``field``.
    delimiter : :class:`str` or :obj:`int`
        The character or position by which to separate ``field``.

    Returns
    -------
    :class:`.Table`
        Table with original ``field`` split into fields whose names are defined
        by `into`."""

    if isinstance(delim, int):
        ht = ht.annotate(**{into[0]: ht[field][:delim], into[1]: ht[field][delim:]})
    else:
        split = ht[field].split(delim)
        ht = ht.annotate(**{into[i]: split[i] for i in range(len(into))})
    ht = ht.drop(field)

    ht_tmp = new_temp_file()
    ht.write(ht_tmp)

    return ht
