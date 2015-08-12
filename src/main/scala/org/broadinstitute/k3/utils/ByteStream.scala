package org.broadinstitute.k3.utils

// FIXME Input
class ByteStream(a: Array[Byte]) {
  require(a != null)
  var i = 0

  def readByte(): Byte = {
    i += 1
    a(i - 1)
  }

  def readULEB128(): Int = {
    var x: Int = 0
    var shift: Int = 0
    do {
      x = x | ((a(i) & 0x7f) << shift)
      i += 1
      shift += 7
    } while ((a(i - 1) & 0x80) != 0)
    x
  }

  def eos = i == a.length
}
