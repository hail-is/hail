package is.hail

import is.hail.annotations.Annotation
import is.hail.utils.{IntIterator, SharedIterable, SharedIterator}

import scala.language.implicitConversions

package object variant {
  type VariantDataset = VariantSampleMatrix[Genotype]
  type GenericDataset = VariantSampleMatrix[Annotation]
  type Call = java.lang.Integer

  class RichSharedIterableGenotype(val ig: SharedIterable[Genotype]) extends AnyVal {
    def toGenotypeStream(v: Variant, isDosage: Boolean): GenotypeStream =
      ig match {
        case gs: GenotypeStream => gs
        case _ =>
          val b: GenotypeStreamBuilder = new GenotypeStreamBuilder(v.nAlleles, isDosage = isDosage)
          b ++= ig.iterator
          b.result()
      }

    def hardCallIterator: IntIterator = ig match {
      case gs: GenotypeStream => gs.gsHardCallIterator
      case _ =>
        new IntIterator {
          val it: SharedIterator[Genotype] = ig.iterator
          override def hasNext: Boolean = it.hasNext
          override def nextInt(): Int = it.next().unboxedGT
        }
    }
  }

  implicit def toRichIterableGenotype(ig: SharedIterable[Genotype]): RichSharedIterableGenotype = new RichSharedIterableGenotype(ig)

  implicit def toVDSFunctions(vds: VariantDataset): VariantDatasetFunctions = new VariantDatasetFunctions(vds)
  implicit def toGDSFunctions(gds: GenericDataset): GenericDatasetFunctions = new GenericDatasetFunctions(gds)
}
