package is.hail.utils

import scala.collection.mutable

object Bitstring {
  def apply(string: String): Bitstring = {
    assert(string.forall(c => c == '0' || c == '1'))
    val bitstring = mutable.ArrayBuilder.make[Long]()
    var pos: Int = 0
    while (string.length - pos > 64) {
      bitstring += java.lang.Long.parseUnsignedLong(string.slice(pos, pos + 64), 2)
      pos += 64
    }
    val lastWord = java.lang.Long.parseUnsignedLong(string.slice(pos, string.length))
    val bitsInLastWord = string.length - pos
    bitstring += (lastWord << (64 - bitsInLastWord))
    new Bitstring(bitstring.result(), bitsInLastWord)
  }
}

case class Bitstring(contents: IndexedSeq[Long], bitsInLastWord: Int) {
  def numWords = contents.length
  def length = (contents.length - 1) * 64 + bitsInLastWord

  override def toString: String = {
    if (contents.isEmpty) return "Bitstring()"
    val result = new mutable.StringBuilder("Bitstring(")
    var i = 0
    while (i < contents.length - 1) {
      result ++= contents(i).toBinaryString
      i += 1
    }
    i = 0
    var lastWord = contents.last
    val bits = Array('0', '1')
    while (i < bitsInLastWord) {
      result += bits((lastWord >>> 63).toInt)
      lastWord <<= 1
      i += 1
    }
    result += ')'
    result.result
  }

  def ++(rhs: Bitstring): Bitstring = {
    if (length == 0) return rhs
    if (rhs.length == 0) return this
    if (bitsInLastWord < 64) {
      val newNumWords = (length + rhs.length + 63) >> 6
      val newContents = Array.ofDim[Long](newNumWords)
      for (i <- 0 until (numWords - 2))
        newContents(i) = contents(i)
      newContents(numWords - 1) = contents.last & (rhs.contents.head >>> bitsInLastWord)
      for (i <- 0 until (rhs.numWords - 2))
        newContents(numWords + i) =
          (rhs.contents(i) << (64 - bitsInLastWord)) &
            (rhs.contents(i + 1) >>> bitsInLastWord)
      var newBitsInLastWord = bitsInLastWord + rhs.bitsInLastWord
      if (newBitsInLastWord > 64) {
        newContents(numWords + rhs.numWords - 1) = rhs.contents.last << (64 - bitsInLastWord)
        newBitsInLastWord = newBitsInLastWord - 64
      }
      new Bitstring(newContents, newBitsInLastWord)
    } else {
      new Bitstring(contents ++ rhs.contents, rhs.bitsInLastWord)
    }
  }

  def popWords(n: Int): (Array[Long], Bitstring) = {
    assert(n < numWords || (n == numWords && bitsInLastWord == 64))
    val result = contents.slice(0, n).toArray
    val newContents = contents.slice(n, numWords)
    val newBitsInLastWord = if (n < numWords) bitsInLastWord else 0
    (result, new Bitstring(newContents, newBitsInLastWord))
  }

  def padTo(n: Int): Array[Long] = {
    assert(n > numWords || (n == numWords && bitsInLastWord < 64))
    val result = Array.ofDim[Long](n)
    Array.copy(contents, 0, result, 0, numWords)
    if (bitsInLastWord == 64) {
      result(numWords) = 1L << 63
    } else {
      result(numWords - 1) = result(numWords - 1) & (1L << (63 - bitsInLastWord))
    }
    result
  }
}
