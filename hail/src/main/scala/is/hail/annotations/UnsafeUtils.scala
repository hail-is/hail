package is.hail.annotations

import is.hail.asm4s._
import is.hail.expr.types.physical.PType

object UnsafeUtils {
  val nativeWordSize: Long = if(System.getProperty("sun.arch.data.model") == "64") 8 else 4

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
}
