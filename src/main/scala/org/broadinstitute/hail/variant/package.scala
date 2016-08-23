package org.broadinstitute.hail

import scala.language.implicitConversions

package object variant {
  type VariantDataset = VariantSampleMatrix[Genotype]

  class RichIterableGenotype(val it: Iterable[Genotype]) extends AnyVal {
    def toGenotypeStream(v: Variant, isDosage: Boolean, compress: Boolean): GenotypeStream =
      it match {
        case gs: GenotypeStream => gs
        case _ =>
          val b: GenotypeStreamBuilder = new GenotypeStreamBuilder(v.nAlleles, isDosage = isDosage, compress = compress)
          b ++= it
          b.result()
      }
  }

  implicit def toRichIterableGenotype(it: Iterable[Genotype]): RichIterableGenotype = new RichIterableGenotype(it)

  implicit def toRichVDS(vsm: VariantDataset): RichVDS = new RichVDS(vsm)
}
