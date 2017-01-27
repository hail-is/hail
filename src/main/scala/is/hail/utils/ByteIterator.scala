package is.hail.utils

final class ByteIterator(val a: Array[Byte]) {
  var i: Int = 0

  def hasNext: Boolean = i < a.length

  def next(): Byte = {
    val r = a(i)
    i += 1
    r
  }

  def skipLEB128(count: Int): Unit = {
    var j = 0
    while (j < count) {
      var b: Byte = next()
      while ((b & 0x80) != 0)
        b = next()
      j += 1
    }
  }

  def readULEB128(): Int = {
    var b: Byte = next()
    var x: Int = b & 0x7f
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = next()
      x |= ((b & 0x7f) << shift)
      shift += 7
    }

    x
  }

  def readSLEB128(): Int = {
    var b: Byte = next()
    var x: Int = b & 0x7f
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = next()
      x |= ((b & 0x7f) << shift)
      shift += 7
    }

    // sign extend
    if (shift < 32
      && (b & 0x40) != 0)
      x = (x << (32 - shift)) >> (32 - shift)

    x
  }
}
