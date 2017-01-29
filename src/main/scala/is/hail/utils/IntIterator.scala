package is.hail.utils

import is.hail.variant.Genotype

abstract class IntIterator extends Iterator[Int] {
  def nextInt(): Int
  override def next(): Int = nextInt()
}

class HardCallIterator(val it: Iterator[Genotype]) extends IntIterator {
  override def hasNext: Boolean = it.hasNext

  override def nextInt(): Int = it.next().unboxedGT
}