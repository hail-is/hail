package is.hail.utils.richUtils

import is.hail.utils.{IntIterator, HardCallIterator}
import is.hail.variant.Genotype

class RichIterableGenotype(it: Iterable[Genotype]) {
  def hardCallIterator: HardCallIterator = new HardCallIterator(it.iterator)
}
