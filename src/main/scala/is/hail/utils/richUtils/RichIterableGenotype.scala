package is.hail.utils.richUtils

import is.hail.utils.IntIterator
import is.hail.variant.{Genotype, GenotypeStream}

class RichIterableGenotype(it: Iterable[Genotype]) {
  def hardCallIterator: IntIterator = it match {
    case gs: GenotypeStream => gs.hardCallIterator
    case _ =>
      val iter = it.iterator
      class HardCallIterator extends IntIterator {
        override def hasNext: Boolean = iter.hasNext
        override def nextInt(): Int = iter.next().unboxedGT
      }
      new HardCallIterator
  }
}
