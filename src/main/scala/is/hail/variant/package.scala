package is.hail

import is.hail.annotations.Annotation
import is.hail.utils.HailIterator

import scala.language.implicitConversions

package object variant {
  type VariantDataset = VariantSampleMatrix[Genotype]
  type GenericDataset = VariantSampleMatrix[Annotation]
  type Call = java.lang.Integer

  class RichIterableGenotype(val ig: Iterable[Genotype]) extends AnyVal {
    def toGenotypeStream(v: Variant, isDosage: Boolean): GenotypeStream =
      ig match {
        case gs: GenotypeStream => gs
        case _ =>
          val b: GenotypeStreamBuilder = new GenotypeStreamBuilder(v.nAlleles, isDosage = isDosage)
          b ++= ig
          b.result()
      }

    def hardCallGenotypeIterator: HailIterator[Int] = ig match {
      case gs: GenotypeStream => gs.gsHardCallGenotypeIterator
      case _ =>
        new HailIterator[Int] {
          val it: Iterator[Genotype] = ig.iterator
          override def hasNext: Boolean = it.hasNext
          override def next(): Int = it.next().unboxedGT
        }
    }

    def biallelicDosageIterator: HailIterator[Double] = ig match {
      case gs: GenotypeStream => gs.gsBiallelicDosageIterator
      case _ =>
        new HailIterator[Double] {
          val it: Iterator[Genotype] = ig.iterator
          override def hasNext: Boolean = it.hasNext
          override def next(): Double = it.next().unboxedBiallelicDosage
        }
    }
  }

  implicit def toRichIterableGenotype(ig: Iterable[Genotype]): RichIterableGenotype = new RichIterableGenotype(ig)

  implicit def toVDSFunctions(vds: VariantDataset): VariantDatasetFunctions = new VariantDatasetFunctions(vds)
  implicit def toGDSFunctions(gds: GenericDataset): GenericDatasetFunctions = new GenericDatasetFunctions(gds)
}
