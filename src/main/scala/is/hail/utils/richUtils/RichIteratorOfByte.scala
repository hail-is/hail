package is.hail.utils.richUtils

class RichIteratorOfByte(val i: Iterator[Byte]) extends AnyVal {
  /*
  def readULEB128(): Int = {
    var x: Int = 0
    var shift: Int = 0
    var b: Byte = 0
    do {
      b = i.next()
      x = x | ((b & 0x7f) << shift)
      shift += 7
    } while ((b & 0x80) != 0)

    x
  }

  def readSLEB128(): Int = {
    var shift: Int = 0
    var x: Int = 0
    var b: Byte = 0
    do {
      b = i.next()
      x |= ((b & 0x7f) << shift)
      shift += 7
    } while ((b & 0x80) != 0)

    // sign extend
    if (shift < 32
      && (b & 0x40) != 0)
      x = (x << (32 - shift)) >> (32 - shift)

    x
  }
  */


}
