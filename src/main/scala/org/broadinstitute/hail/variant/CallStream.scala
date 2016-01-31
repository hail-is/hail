package org.broadinstitute.hail.variant

import org.broadinstitute.hail.variant.CallStream.CallStream

object CallStream {
  type CallStream = Iterable[(Int, Int)]

  def toBinary(b: Byte): String = {
    for (i <- 7 to 0 by -1) yield (b & (1 << i)) >> i
  }.mkString("")


  def byteToInts(b: Byte): String = {
    for (i <- 6 to 0 by -2) yield (b & (3 << i)) >> i
  }.mkString(":")

  def encodeGtByte(gts: Array[Int], s: Int): Byte =
    (if (s + 3 < gts.length)
      gts(s) | gts(s + 1) << 2 | gts(s + 2) << 4 | gts(s + 3) << 6
    else if (s + 3 == gts.length)
      gts(s) | gts(s + 1) << 2 | gts(s + 2) << 4
    else if (s + 2 == gts.length)
      gts(s) | gts(s + 1) << 2
    else
      gts(s)
      ).toByte
}

object DenseCalls {

  def apply(gts: Array[Int]): DenseCalls = {
    val nBytes = (gts.length + 3) >> 2
    val a = Array.ofDim[Byte](nBytes)

//    a.indices.foreach(j => a(j) = encodeByte(gts, 4 * j)) FIXME: would this be faster or slower?

    for (i <- 0 until nBytes)
      a(i) = CallStream.encodeGtByte(gts, i << 2)

    DenseCalls(a)
  }

  // I wrote out each case to optimize speed...the first occurs 99.9% of the time, so didn't want to build up long expression

  def decodeByte(b: Byte, s: Int): CallStream =
    for {
      j <- 0 until 4
      gt = (b & (3 << (2 * j))) >> (2 * j)
      if gt != 0
    } yield (s + j, gt)
}

case class DenseCalls(a: Array[Byte]) extends CallStream {
  def iterator =
    a.zipWithIndex
      .flatMap{ case (b, i) => DenseCalls.decodeByte(b, 4 * i)}
      .iterator

  def showBinary() = println(a.map(b => CallStream.toBinary(b)).mkString("[", ", ", "]"))

  override def toString() = a.map(b => CallStream.byteToInts(b)).mkString("[", ", ", "]")
}

object SparseCalls {
  def apply(gts: Array[Int]): SparseCalls = {
    SparseCalls(Array[Byte]())
  }

  def encodeBytes(sparseGts: Array[(Int, Int)]): Iterator[Byte] = {
    val gtByte = CallStream.encodeGtByte(sparseGts.map(_._2), 0)
    val lByte = encodeLByte(sparseGts.map(_._1))
    val sBytes = Iterator(0)
  }

  def encodeLByte(ss: Array[Int]): Byte = ss.map(nBytesForInt).map(CallStream.encodeGtByte)
}

case class SparseCalls(a: Array[Byte]) extends CallStream {
  def iterator = Iterator()
}