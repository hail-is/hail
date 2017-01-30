package is.hail.utils.richUtils

import is.hail.utils.IntIterator
import is.hail.variant.{Genotype, GenotypeStream}

class RichIterableGenotype(gs: Iterable[Genotype]) {
  def hardCallIterator: IntIterator = gs match {
    case gs: GenotypeStream => gs.gsHardCallIterator
    case _ =>
      new IntIterator {
        val it: Iterator[Genotype] = gs.iterator
        override def hasNext: Boolean = it.hasNext
        override def nextInt(): Int = it.next().unboxedGT
      }
  }
}