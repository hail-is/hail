package is.hail.annotations

import is.hail.expr.Type

object UnsafeUtils {
  def arrayElementSize(t: Type): Long = roundUpAlignment(t.byteSize, t.alignment)

  def roundUpAlignment(offset: Long, alignment: Long): Long = {
    assert(alignment > 0, s"invalid alignment: $alignment")
    assert((alignment & (alignment - 1)) == 0, s"invalid alignment: $alignment") // power of 2
    (offset + (alignment - 1)) & ~(alignment - 1)
  }
}
