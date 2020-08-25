Relatedness
-----------

.. currentmodule:: hail.methods

The *relatedness* of two individuals characterizes their biological
relationship. For example, two individuals might be siblings or
parent-and-child. All notions of relatedness implemented in Hail are rooted in
the idea of alleles "inherited identically by descent". Two alleles in two
distinct individuals are inherited identically by descent if both alleles were
inherited by the same "recent," common ancestor. The term "recent" distinguishes
alleles shared IBD from family members from alleles shared IBD from "distant"
ancestors. Distant ancestors are thought of contributing to population structure
rather than relatedness.

Relatedness is usually quantified by two quantities: kinship coefficient
(:math:`\phi` or ``PI_HAT``) and probability-of-identity-by-descent-zero
(:math:`\pi_0` or ``Z0``). The kinship coefficient is the probability that any
two alleles selected randomly from the same locus are identical by
descent. Twice the kinship coefficient is the coefficient of relationship which
is the percent of genetic material shared identically by descent.
Probability-of-identity-by-descent-zero is the probability that none of the
alleles at a randomly chosen locus were inherited identically by descent.

Hail provides three methods for the inference of relatedness: PLINK-style
identity by descent, KING, and PC-Relate.

- :func:`.identity_by_descent` is appropriate for datasets containing one
  homogeneous population.
- :func:`.king` is appropriate for datasets containing multiple homogeneous
  populations and no admixture. It is also used to prune close relatives before
  using :func:`.pc_relate`.
- :func:`.pc_relate` is appropriate for datasets containing multiple
  homogeneous populations and admixture.

.. toctree::
    :maxdepth: 2

.. autosummary::

    identity_by_descent
    pc_relate
    king

.. autofunction:: identity_by_descent
.. autofunction:: pc_relate
.. autofunction:: king
