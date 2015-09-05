package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant.{Genotype, Variant}

object nDeletionPerSample extends SumMethod {
  def name = "nDeletion"
  override def mapWithKeys(v: Variant, s: Int, g: Genotype) =
    if (g.isNonRef && v.isDeletion) 1 else 0
}

object nInsertionPerSample extends SumMethod {
  def name = "nInsertion"
  override def mapWithKeys(v: Variant, s: Int, g: Genotype) =
    if (g.isNonRef && v.isInsertion) 1 else 0
}

object rDeletionInsertionPerSample extends DerivedMethod {
  type T = Double
  def name = "rDeletionInsertion"
  def map(values: MethodValues) = {
    val nDel = values.get(nDeletionPerSample)
    val nIns = values.get(nInsertionPerSample)
    if (nIns != 0) nDel.toDouble / nIns else -1
  }
}

object nTransversionPerSample extends SumMethod {
  def name = "nTransversion"
  override def mapWithKeys(v: Variant, s: Int, g: Genotype) =
    if (g.isNonRef && v.isTransversion) 1 else 0
}

object nTransitionPerSample extends SumMethod {
  def name = "nTransition"
  override def mapWithKeys(v: Variant, s: Int, g: Genotype) =
    if (g.isNonRef && v.isTransition) 1 else 0
}

object rTiTvPerSample extends DerivedMethod {
  type T = Double
  def name = "rTiTv"
  def map(values: MethodValues) = {
    val nTi = values.get(nTransitionPerSample)
    val nTv = values.get(nTransversionPerSample)
    if (nTv != 0) nTi.toDouble / nTv else -1
  }
}

object nSNPPerSample extends SumMethod {
  def name = "nSNP"
  override def mapWithKeys(v: Variant, s: Int, g: Genotype) = if (g.isNonRef && v.isSNP) 1 else 0
}
