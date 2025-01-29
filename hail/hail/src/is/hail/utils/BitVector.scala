package is.hail.utils

import java.util

final class BitVector(val length: Int) {
  private val a: Array[Long] = new Array[Long]((length + 63) / 64)

  def size: Int = length

  def apply(i: Int): Boolean = {
    if (i < 0 || i >= length)
      throw new ArrayIndexOutOfBoundsException

    (a(i / 64) & (1L << (i & 63))) != 0
  }

  def set(i: Int): Unit = {
    if (i < 0 || i >= length)
      throw new ArrayIndexOutOfBoundsException

    a(i / 64) |= (1L << (i & 63))
  }

  def reset(i: Int): Unit = {
    if (i < 0 || i >= length)
      throw new ArrayIndexOutOfBoundsException

    a(i / 64) &= ~(1L << (i & 63))
  }

  def clear(): Unit =
    util.Arrays.fill(a, 0)
}
