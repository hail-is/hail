package org.broadinstitute.hail

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.variant.{GenotypeStream, Variant}

package object variant {
  type VariantDataset = VariantSampleMatrix[Genotype]

  // type VariantSampleMatrix[T, S] = managed.ManagedVSM[T, S]
  // type VariantSampleMatrix[T, S <: Iterable[(Int, T)]] = sparky.SparkyVSM[T, S]
  // def importToVSM(rdd: RDD[(Variant, GenotypeStream)]) = rdd

  // type VariantSampleMatrix[T, S] = tuple.TupleVSM[T]

  // def importToVSM(rdd: RDD[(Variant, GenotypeStream)]): RDD[(Variant, Int, Genotype)] =
  //   rdd.flatMap { case (v, gs) => gs.iterator.map { case (s, g) => (v, s, g) } }
}
