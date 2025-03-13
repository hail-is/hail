package is.hail.utils

import java.util

object BufferedAggregatorIterator {
  val loadFactor: Float = 0.75f
}

class BufferedAggregatorIterator[T, V, U, K](
  it: Iterator[T],
  makeComb: () => V,
  makeKey: T => K,
  sequence: (T, V) => Unit,
  serializeAndCleanup: V => U,
  bufferSize: Int,
) extends Iterator[(K, U)] {

  private val fb = it.toFlipbookIterator
  private var popped: util.Map.Entry[K, V] = _

  // TODO: use a heavy hitters buffer here
  private val buffer = new util.LinkedHashMap[K, V](
    (bufferSize / BufferedAggregatorIterator.loadFactor).toInt + 1,
    BufferedAggregatorIterator.loadFactor,
    true,
  ) {
    override def removeEldestEntry(eldest: util.Map.Entry[K, V]): Boolean =
      if (size() > bufferSize) {
        popped = eldest
        true
      } else false
  }

  def hasNext: Boolean =
    fb.isValid || buffer.size() > 0

  def next(): (K, U) = {
    if (!hasNext)
      throw new NoSuchElementException
    while (fb.isValid) {
      val value = fb.value
      val key = makeKey(value)
      if (buffer.containsKey(key)) {
        sequence(value, buffer.get(key))
        fb.advance()
      } else {
        val agg = makeComb()
        sequence(value, agg)
        fb.advance()
        buffer.put(key, agg)
        if (popped != null) {
          val cp = popped
          popped = null
          return cp.getKey -> serializeAndCleanup(cp.getValue)
        }
      }
    }
    val next = buffer.entrySet().iterator().next()
    buffer.remove(next.getKey)
    next.getKey -> serializeAndCleanup(next.getValue)
  }
}
