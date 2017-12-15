package is.hail

import is.hail.annotations.Annotation
import is.hail.utils.HailIterator

import scala.language.implicitConversions

package object variant {
  type Call = java.lang.Integer

  class RichIterableGenotype(val ig: Iterable[Annotation]) extends AnyVal {
    def hardCallIterator: HailIterator[Int] =
      new HailIterator[Int] {
        val it: Iterator[Annotation] = ig.iterator

        override def hasNext: Boolean = it.hasNext

        override def next(): Int = {
          val g = it.next()
          Genotype.unboxedGT(g)
        }
      }
  }

  implicit def toRichIterableGenotype(ig: Iterable[Annotation]): RichIterableGenotype = new RichIterableGenotype(ig)

  implicit def toVDSFunctions(vds: VariantSampleMatrix): VariantDatasetFunctions = new VariantDatasetFunctions(vds)
}
