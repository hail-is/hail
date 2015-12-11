package org.broadinstitute.hail

import org.apache.spark.rdd.RDD
import scala.language.implicitConversions
import org.broadinstitute.hail.variant.{GenotypeStream, Variant}

package object variant {
  type VariantDataset = VariantSampleMatrix[Genotype]

  class RichIterableGenotype(val it: Iterable[Genotype]) extends AnyVal {
    def toGenotypeStream(v: Variant, compress: Boolean): GenotypeStream =
      it match {
        case gs: GenotypeStream => gs
        case _ =>
          val b: GenotypeStreamBuilder = new GenotypeStreamBuilder(v, compress = compress)
          b ++= it
          b.result()
      }
  }

  implicit def toRichIterableGenotype(it: Iterable[Genotype]): RichIterableGenotype = new RichIterableGenotype(it)
  implicit def toRichVDS(vsm: VariantDataset): RichVDS = new RichVDS(vsm)

  // type VariantSampleMatrix[T, S] = managed.ManagedVSM[T, S]
  // type VariantSampleMatrix[T, S <: Iterable[(Int, T)]] = sparky.SparkyVSM[T, S]
  // def importToVSM(rdd: RDD[(Variant, GenotypeStream)]) = rdd

  // type VariantSampleMatrix[T, S] = tuple.TupleVSM[T]

  // def importToVSM(rdd: RDD[(Variant, GenotypeStream)]): RDD[(Variant, Int, Genotype)] =
  //   rdd.flatMap { case (v, gs) => gs.iterator.map { case (s, g) => (v, s, g) } }
}
