package is.hail.annotations

import is.hail.asm4s._
import is.hail.types.physical.PType

object UnsafeUtils {
  def arrayElementSize(t: PType): Long = roundUpAlignment(t.byteSize, t.alignment)

  def roundUpAlignment(offset: Long, alignment: Long): Long = {
    assert(alignment > 0)
    assert((alignment & (alignment - 1)) == 0) // power of 2
    (offset + (alignment - 1)) & ~(alignment - 1)
  }

  def roundUpAlignment(offset: Code[Long], alignment: Long): Code[Long] = {
    assert(alignment > 0)
    assert((alignment & (alignment - 1)) == 0) // power of 2
    (offset + (alignment - 1)) & ~(alignment - 1)
  }

  def roundDownAlignment(offset: Long, alignment: Long): Long = {
    assert(alignment > 0)
    assert((alignment & (alignment - 1)) == 0) // power of 2
    offset & -alignment
  }

  def roundDownAlignment(offset: Code[Long], alignment: Long): Code[Long] = {
    assert(alignment > 0)
    assert((alignment & (alignment - 1)) == 0) // power of 2
    offset & -alignment
  }

  def roundDownAlignment(offset: Int, alignment: Int): Int = {
    assert(alignment > 0)
    assert((alignment & (alignment - 1)) == 0) // power of 2
    offset & -alignment
  }

  def roundDownAlignment(offset: Code[Int], alignment: Int): Code[Int] = {
    assert(alignment > 0)
    assert((alignment & (alignment - 1)) == 0) // power of 2
    offset & -alignment
  }

  def packBitsToBytes(nBits: Int): Int =
    (nBits + 7) >>> 3

  def packBitsToBytes(nBits: Code[Int]): Code[Int] =
    (nBits + 7) >>> 3
}
