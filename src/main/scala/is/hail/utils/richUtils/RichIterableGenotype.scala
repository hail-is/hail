package is.hail.utils.richUtils

import is.hail.utils.{IntIterator, GenericIntIterator}
import is.hail.variant.Genotype

class RichIterableGenotype(it: Iterable[Genotype]) {
  def hardcallIterator: IntIterator = new GenericIntIterator(it.iterator.map(_.unboxedGT))
}
