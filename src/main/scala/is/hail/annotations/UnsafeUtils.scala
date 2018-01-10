package is.hail.annotations

import is.hail.asm4s._
import is.hail.expr.types.Type

object UnsafeUtils {
  def arrayElementSize(t: Type): Long = roundUpAlignment(t.byteSize, t.alignment)

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
}
