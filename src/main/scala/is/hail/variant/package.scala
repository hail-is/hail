package is.hail

import is.hail.utils.IntIterator

import scala.language.implicitConversions

package object variant {
  type VariantDataset = VariantSampleMatrix[Genotype]

  class RichIterableGenotype(val git: Iterable[Genotype]) extends AnyVal {
    def toGenotypeStream(v: Variant, isDosage: Boolean, compress: Boolean): GenotypeStream =
      git match {
        case gs: GenotypeStream => gs
        case _ =>
          val b: GenotypeStreamBuilder = new GenotypeStreamBuilder(v.nAlleles, isDosage = isDosage, compress = compress)
          b ++= git
          b.result()
      }

    def hardCallIterator: IntIterator = git match {
      case gs: GenotypeStream => gs.gsHardCallIterator
      case _ =>
        new IntIterator {
          val it: Iterator[Genotype] = git.iterator
          override def hasNext: Boolean = it.hasNext
          override def nextInt(): Int = it.next().unboxedGT
        }
    }
  }

  implicit def toRichIterableGenotype(it: Iterable[Genotype]): RichIterableGenotype = new RichIterableGenotype(it)

  implicit def toRichVDS(vsm: VariantDataset): RichVDS = new RichVDS(vsm)
}
