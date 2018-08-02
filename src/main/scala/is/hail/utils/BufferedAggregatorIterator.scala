package is.hail.utils

import java.util

import is.hail.annotations.aggregators.{RegionValueAggregator, RegionValueCountAggregator}

class BufferedAggregatorIterator[T, U, K](
  it: Iterator[T],
  makeComb: () => U,
  makeKey: T => K,
  sequence: (T, U) => Unit,
  bufferSize: Int
) extends Iterator[(K, U)] {

  private val buffer = new Array[(K, U)](bufferSize)
  private val keyIdx: util.HashMap[K, Int] = new util.HashMap()
  private var ptr = 0

  private def bump(idx: Int): Unit = {
    if (idx > 0) {
      val curr = buffer(idx)
      val prev = buffer(idx - 1)
      keyIdx.put(curr._1, idx - 1)
      keyIdx.put(prev._1, idx)
      buffer(idx) = prev
      buffer(idx - 1) = curr
    }
  }

  private def fill() {
    while (ptr < bufferSize && it.hasNext) {
      val rv = it.next()
      val key = makeKey(rv)
      if (keyIdx.containsKey(key)) {
        val idx = keyIdx.get(key)
        sequence(rv, buffer(idx)._2)
        bump(idx)
      } else {
        val agg = makeComb()
        sequence(rv, agg)
        keyIdx.put(key, ptr)
        buffer(ptr) = key -> agg
        ptr += 1
      }
    }
  }

  def hasNext: Boolean = {
    fill()
    ptr != 0
  }

  def next(): (K, U) = {
    if (!hasNext)
      throw new NoSuchElementException
    val last = buffer(ptr - 1)
    keyIdx.remove(last._1)
    ptr -= 1
    last
  }
}
