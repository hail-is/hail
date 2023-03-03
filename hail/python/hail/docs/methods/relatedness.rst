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
identity by descent [1]_, KING [2]_, and PC-Relate [3]_.

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
    king
    pc_relate
    simulate_random_mating

.. autofunction:: identity_by_descent
.. autofunction:: king
.. autofunction:: pc_relate
.. autofunction:: simulate_random_mating

.. [1] Purcell, Shaun et al. “PLINK: a tool set for whole-genome association and
       population-based linkage analyses.” American journal of human genetics
       vol. 81,3 (2007):
       559-75. doi:10.1086/519795. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1950838/
.. [2] Manichaikul, Ani et al. “Robust relationship inference in genome-wide
       association studies.” Bioinformatics (Oxford, England) vol. 26,22 (2010):
       2867-73. doi:10.1093/bioinformatics/btq559. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3025716/
.. [3] Conomos, Matthew P et al. “Model-free Estimation of Recent Genetic
       Relatedness.” American journal of human genetics vol. 98,1 (2016):
       127-48. doi:10.1016/j.ajhg.2015.11.022. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4716688/
