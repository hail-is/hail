package is.hail.annotations

import is.hail.expr.Type

object UnsafeUtils {
  def arrayElementSize(t: Type): Int = {
    var eltSize = t.byteSize
    if (eltSize > 0) {
      val mod = eltSize % t.alignment
      if (mod != 0)
        eltSize += (t.alignment - mod)
    }
    eltSize
  }

  def roundUpAlignment(offset: Int, alignment: Int): Int = {
    assert(alignment > 0, s"invalid alignment: $alignment")
    assert((alignment & (alignment - 1)) == 0, s"invalid alignment: $alignment") // power of 2
    (offset + (alignment - 1)) & ~(alignment - 1)
  }
}
