package is.hail

import is.hail.annotations.Annotation
import is.hail.utils.HailIterator

import scala.language.implicitConversions

package object variant {
  type VariantDataset = VariantSampleMatrix[GenotypeMatrixT]
  type GenericDataset = VariantSampleMatrix[GenericMatrixT]
  type Call = java.lang.Integer

  class RichIterableGenotype(val ig: Iterable[Genotype]) extends AnyVal {
    def toGenotypeStream(v: Variant, isLinearScale: Boolean): GenotypeStream =
      ig match {
        case gs: GenotypeStream => gs
        case _ =>
          if (ig.isEmpty)
            GenotypeStream.empty(v.nAlleles)
          else {
            val b: GenotypeStreamBuilder = new GenotypeStreamBuilder(v.nAlleles, isLinearScale = isLinearScale)
            b ++= ig
            b.result()
          }
      }

    def hardCallIterator: HailIterator[Int] = ig match {
      case gs: GenotypeStream => gs.gsHardCallIterator
      case _ =>
        new HailIterator[Int] {
          val it: Iterator[Genotype] = ig.iterator
          override def hasNext: Boolean = it.hasNext
          override def next(): Int = it.next().unboxedGT
        }
    }

    def dosageIterator: HailIterator[Double] = ig match {
      case gs: GenotypeStream => gs.gsDosageIterator
      case _ =>
        new HailIterator[Double] {
          val it: Iterator[Genotype] = ig.iterator
          override def hasNext: Boolean = it.hasNext
          override def next(): Double = it.next().unboxedDosage
        }
    }
  }

  implicit def toRichIterableGenotype(ig: Iterable[Genotype]): RichIterableGenotype = new RichIterableGenotype(ig)

  implicit def toVDSFunctions(vds: VariantDataset): VariantDatasetFunctions = new VariantDatasetFunctions(vds)
  implicit def toGDSFunctions(gds: GenericDataset): GenericDatasetFunctions = new GenericDatasetFunctions(gds)
}
