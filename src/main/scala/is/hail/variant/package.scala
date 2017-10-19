package is.hail

import is.hail.annotations.Annotation
import is.hail.utils.HailIterator

import scala.language.implicitConversions

package object variant {
  type VariantDataset = VariantSampleMatrix[Locus, Variant, Genotype]
  type VariantKeyDataset = VariantSampleMatrix[Locus, Variant, Annotation]
  type GenericDataset = VariantSampleMatrix[Annotation, Annotation, Annotation]
  type Call = java.lang.Integer

  class RichIterableGenotype(val ig: Iterable[Genotype]) extends AnyVal {
    def hardCallIterator: HailIterator[Int] =
      new HailIterator[Int] {
        val it: Iterator[Genotype] = ig.iterator

        override def hasNext: Boolean = it.hasNext

        override def next(): Int = {
          val g = it.next()
          Genotype.unboxedGT(g)
        }
      }

    def dosageIterator: HailIterator[Double] =
      new HailIterator[Double] {
        val it: Iterator[Genotype] = ig.iterator

        override def hasNext: Boolean = it.hasNext

        override def next(): Double =
          Genotype.unboxedDosage(it.next())
      }
  }

  implicit def toRichIterableGenotype(ig: Iterable[Genotype]): RichIterableGenotype = new RichIterableGenotype(ig)

  implicit def toVDSFunctions(vds: VariantDataset): VariantDatasetFunctions = new VariantDatasetFunctions(vds)

  implicit def toVKDSFunctions[T >: Null](vkds: VariantSampleMatrix[Locus, Variant, T]): VariantKeyDatasetFunctions[T] = new VariantKeyDatasetFunctions(vkds)
}
