from typing import List

import hail as hl
from hail.expr.expressions import expr_array, expr_call, expr_locus, expr_str
from hail.typecheck import sequenceof, typecheck


@typecheck(
    locus=expr_locus(),
    alleles=expr_array(expr_str),
    proband_call=expr_call,
    father_call=expr_call,
    mother_call=expr_call,
)
def phase_by_transmission(
    locus: hl.expr.LocusExpression,
    alleles: hl.expr.ArrayExpression,
    proband_call: hl.expr.CallExpression,
    father_call: hl.expr.CallExpression,
    mother_call: hl.expr.CallExpression,
) -> hl.expr.ArrayExpression:
    """Phases genotype calls in a trio based allele transmission.

    Notes
    -----
    In the phased calls returned, the order is as follows:
    - Proband: father_allele | mother_allele
    - Parents: transmitted_allele | untransmitted_allele

    Phasing of sex chromosomes:
    - Sex chromosomes of male individuals should be haploid to be phased correctly.
    - If `proband_call` is diploid on non-par regions of the sex chromosomes, it is assumed to be female.

    Returns `NA` when genotype calls cannot be phased.
    The following genotype calls combinations cannot be phased by transmission:
    1. One of the calls in the trio is missing
    2. The proband genotype cannot be obtained from the parents alleles (Mendelian violation)
    3. All individuals of the trio are heterozygous for the same two alleles
    4. Father is diploid on non-PAR region of X or Y
    5. Proband is diploid on non-PAR region of Y

    In addition, individual phased genotype calls are returned as missing in the following situations:
    1. All mother genotype calls non-PAR region of Y
    2. Diploid father genotype calls on non-PAR region of X for a male proband (proband and mother are still phased as father doesn't participate in allele transmission)

    Note
    ----
    :func:`~.phase_trio_matrix_by_transmission` provides a convenience wrapper for phasing a trio matrix.

    Parameters
    ----------
    locus : :class:`.LocusExpression`
        Expression for the locus in the trio matrix
    alleles : :class:`.ArrayExpression`
        Expression for the alleles in the trio matrix
    proband_call : :class:`.CallExpression`
        Expression for the proband call in the trio matrix
    father_call : :class:`.CallExpression`
        Expression for the father call in the trio matrix
    mother_call : :class:`.CallExpression`
        Expression for the mother call in the trio matrix

    Returns
    -------
    :class:`.ArrayExpression`
        Array containing: [phased proband call, phased father call, phased mother call]"""

    def call_to_one_hot_alleles_array(
        call: hl.expr.CallExpression, alleles: hl.expr.ArrayExpression
    ) -> hl.expr.ArrayExpression:
        """
        Get the set of all different one-hot-encoded allele-vectors in a genotype call.
        It is returned as an ordered array where the first vector corresponds to the first allele,
        and the second vector (only present if het) the second allele.

        :param CallExpression call: genotype
        :param ArrayExpression alleles: Alleles at the site
        :return: Array of one-hot-encoded alleles
        :rtype: ArrayExpression
        """
        return hl.if_else(
            call.is_het(),
            hl.array([
                hl.call(call[0]).one_hot_alleles(alleles),
                hl.call(call[1]).one_hot_alleles(alleles),
            ]),
            hl.array([hl.call(call[0]).one_hot_alleles(alleles)]),
        )

    def phase_parent_call(call: hl.expr.CallExpression, transmitted_allele_index: int):
        """
        Given a genotype and which allele was transmitted to the offspring, returns the parent phased genotype.

        :param CallExpression call: Parent genotype
        :param int transmitted_allele_index: index of transmitted allele (0 or 1)
        :return: Phased parent genotype
        :rtype: CallExpression
        """
        return hl.call(call[transmitted_allele_index], call[hl.int(transmitted_allele_index == 0)], phased=True)

    def phase_diploid_proband(
        locus: hl.expr.LocusExpression,
        alleles: hl.expr.ArrayExpression,
        proband_call: hl.expr.CallExpression,
        father_call: hl.expr.CallExpression,
        mother_call: hl.expr.CallExpression,
    ) -> hl.expr.ArrayExpression:
        """
        Returns phased genotype calls in the case of a diploid proband
        (autosomes, PAR regions of sex chromosomes or non-PAR regions of a female proband)

        :param LocusExpression locus: Locus in the trio MatrixTable
        :param ArrayExpression alleles: Alleles in the trio MatrixTable
        :param CallExpression proband_call: Input proband genotype call
        :param CallExpression father_call: Input father genotype call
        :param CallExpression mother_call: Input mother genotype call
        :return: Array containing: phased proband call, phased father call, phased mother call
        :rtype: ArrayExpression
        """

        proband_v = proband_call.one_hot_alleles(alleles)
        father_v = hl.if_else(
            locus.in_x_nonpar() | locus.in_y_nonpar(),
            hl.or_missing(father_call.is_haploid(), hl.array([father_call.one_hot_alleles(alleles)])),
            call_to_one_hot_alleles_array(father_call, alleles),
        )
        mother_v = call_to_one_hot_alleles_array(mother_call, alleles)

        combinations = hl.flatmap(
            lambda f: hl.enumerate(mother_v)
            .filter(lambda m: m[1] + f[1] == proband_v)
            .map(lambda m: hl.struct(m=m[0], f=f[0])),
            hl.enumerate(father_v),
        )

        return hl.or_missing(
            hl.is_defined(combinations) & (hl.len(combinations) == 1),
            hl.array([
                hl.call(father_call[combinations[0].f], mother_call[combinations[0].m], phased=True),
                hl.if_else(
                    father_call.is_haploid(),
                    hl.call(father_call[0], phased=True),
                    phase_parent_call(father_call, combinations[0].f),
                ),
                phase_parent_call(mother_call, combinations[0].m),
            ]),
        )

    def phase_haploid_proband_x_nonpar(
        proband_call: hl.expr.CallExpression, father_call: hl.expr.CallExpression, mother_call: hl.expr.CallExpression
    ) -> hl.expr.ArrayExpression:
        """
        Returns phased genotype calls in the case of a haploid proband in the non-PAR region of X

        :param CallExpression proband_call: Input proband genotype call
        :param CallExpression father_call: Input father genotype call
        :param CallExpression mother_call: Input mother genotype call
        :return: Array containing: phased proband call, phased father call, phased mother call
        :rtype: ArrayExpression
        """

        transmitted_allele = hl.enumerate(hl.array([mother_call[0], mother_call[1]])).find(
            lambda m: m[1] == proband_call[0]
        )
        return hl.or_missing(
            hl.is_defined(transmitted_allele),
            hl.array([
                hl.call(proband_call[0], phased=True),
                hl.or_missing(father_call.is_haploid(), hl.call(father_call[0], phased=True)),
                phase_parent_call(mother_call, transmitted_allele[0]),
            ]),
        )

    def phase_y_nonpar(
        proband_call: hl.expr.CallExpression,
        father_call: hl.expr.CallExpression,
    ) -> hl.expr.ArrayExpression:
        """
        Returns phased genotype calls in the non-PAR region of Y (requires both father and proband to be haploid to return phase)

        :param CallExpression proband_call: Input proband genotype call
        :param CallExpression father_call: Input father genotype call
        :return: Array containing: phased proband call, phased father call, phased mother call
        :rtype: ArrayExpression
        """
        return hl.or_missing(
            proband_call.is_haploid() & father_call.is_haploid() & (father_call[0] == proband_call[0]),
            hl.array([
                hl.call(proband_call[0], phased=True),
                hl.call(father_call[0], phased=True),
                hl.missing(hl.tcall),
            ]),
        )

    return (
        hl.case()
        .when(
            locus.in_x_nonpar() & proband_call.is_haploid(),
            phase_haploid_proband_x_nonpar(proband_call, father_call, mother_call),
        )
        .when(locus.in_y_nonpar(), phase_y_nonpar(proband_call, father_call))
        .when(proband_call.is_diploid(), phase_diploid_proband(locus, alleles, proband_call, father_call, mother_call))
        .or_missing()
    )


@typecheck(tm=hl.MatrixTable, call_field=str, phased_call_field=str)
def phase_trio_matrix_by_transmission(
    tm: hl.MatrixTable, call_field: str = 'GT', phased_call_field: str = 'PBT_GT'
) -> hl.MatrixTable:
    """Adds a phased genoype entry to a trio MatrixTable based allele transmission in the trio.

    Example
    -------
    >>> # Create a trio matrix
    >>> pedigree = hl.Pedigree.read('data/case_control_study.fam')
    >>> trio_dataset = hl.trio_matrix(dataset, pedigree, complete_trios=True)

    >>> # Phase trios by transmission
    >>> phased_trio_dataset = phase_trio_matrix_by_transmission(trio_dataset)

    Notes
    -----
    Uses only a `Call` field to phase and only phases when all 3 members of the trio are present and have a call.

    In the phased genotypes, the order is as follows:
    - Proband: father_allele | mother_allele
    - Parents: transmitted_allele | untransmitted_allele

    Phasing of sex chromosomes:
    - Sex chromosomes of male individuals should be haploid to be phased correctly.
    - If a proband is diploid on non-par regions of the sex chromosomes, it is assumed to be female.

    Genotypes that cannot be phased are set to `NA`.
    The following genotype calls combinations cannot be phased by transmission (all trio members phased calls set to missing):
    1. One of the calls in the trio is missing
    2. The proband genotype cannot be obtained from the parents alleles (Mendelian violation)
    3. All individuals of the trio are heterozygous for the same two alleles
    4. Father is diploid on non-PAR region of X or Y
    5. Proband is diploid on non-PAR region of Y

    In addition, individual phased genotype calls are returned as missing in the following situations:
    1. All mother genotype calls non-PAR region of Y
    2. Diploid father genotype calls on non-PAR region of X for a male proband (proband and mother are still phased as father doesn't participate in allele transmission)

    Parameters
    ----------
    tm : :class:`.MatrixTable`
        Trio MatrixTable (entries have to be a Struct with `proband_entry`, `mother_entry` and `father_entry` present)
    call_field : str
        genotype field name in the matrix entries to use for phasing
    phased_call_field : str
        name for the phased genotype field in the matrix entries

    Returns
    -------
    :class:`.MatrixTable`
        Trio MatrixTable entry with additional phased genotype field for each individual"""

    tm = tm.annotate_entries(
        __phased_GT=phase_by_transmission(
            tm.locus, tm.alleles, tm.proband_entry[call_field], tm.father_entry[call_field], tm.mother_entry[call_field]
        )
    )

    return tm.select_entries(
        proband_entry=hl.struct(**tm.proband_entry, **{phased_call_field: tm.__phased_GT[0]}),
        father_entry=hl.struct(**tm.father_entry, **{phased_call_field: tm.__phased_GT[1]}),
        mother_entry=hl.struct(**tm.mother_entry, **{phased_call_field: tm.__phased_GT[2]}),
    )


@typecheck(tm=hl.MatrixTable, col_keys=sequenceof(str), keep_trio_cols=bool, keep_trio_entries=bool)
def explode_trio_matrix(
    tm: hl.MatrixTable, col_keys: List[str] = ['s'], keep_trio_cols: bool = True, keep_trio_entries: bool = False
) -> hl.MatrixTable:
    """Splits a trio MatrixTable back into a sample MatrixTable.

    Example
    -------
    >>> # Create a trio matrix from a sample matrix
    >>> pedigree = hl.Pedigree.read('data/case_control_study.fam')
    >>> trio_dataset = hl.trio_matrix(dataset, pedigree, complete_trios=True)

    >>> # Explode trio matrix back into a sample matrix
    >>> exploded_trio_dataset = explode_trio_matrix(trio_dataset)

    Notes
    -----
    The resulting MatrixTable column schema is the same as the proband/father/mother schema,
    and the resulting entry schema is the same as the proband_entry/father_entry/mother_entry schema.
    If the `keep_trio_cols` option is set, then an additional `source_trio` column is added with the trio column data.
    If the `keep_trio_entries` option is set, then an additional `source_trio_entry` column is added with the trio entry data.

    Note
    ----
    This assumes that the input MatrixTable is a trio MatrixTable (similar to
    the result of :func:`~.trio_matrix`) Its entry schema has to contain
    'proband_entry`, `father_entry` and `mother_entry` all with the same type.
    Its column schema has to contain 'proband`, `father` and `mother` all with
    the same type.

    Parameters
    ----------
    tm : :class:`.MatrixTable`
        Trio MatrixTable (entries have to be a Struct with `proband_entry`, `mother_entry` and `father_entry` present)
    col_keys : :obj:`list` of str
        Column key(s) for the resulting sample MatrixTable
    keep_trio_cols: bool
        Whether to add a `source_trio` column with the trio column data (default `True`)
    keep_trio_entries: bool
        Whether to add a `source_trio_entries` column with the trio entry data (default `False`)

    Returns
    -------
    :class:`.MatrixTable`
        Sample MatrixTable
    """

    select_entries_expr = {'__trio_entries': hl.array([tm.proband_entry, tm.father_entry, tm.mother_entry])}
    if keep_trio_entries:
        select_entries_expr['source_trio_entry'] = hl.struct(**tm.entry)
    tm = tm.select_entries(**select_entries_expr)

    tm = tm.key_cols_by()
    select_cols_expr = {'__trio_members': hl.enumerate(hl.array([tm.proband, tm.father, tm.mother]))}
    if keep_trio_cols:
        select_cols_expr['source_trio'] = hl.struct(**tm.col)
    tm = tm.select_cols(**select_cols_expr)

    mt = tm.explode_cols(tm.__trio_members)

    mt = mt.transmute_entries(**mt.__trio_entries[mt.__trio_members[0]])

    mt = mt.key_cols_by()
    mt = mt.transmute_cols(**mt.__trio_members[1])

    if col_keys:
        mt = mt.key_cols_by(*col_keys)

    return mt
