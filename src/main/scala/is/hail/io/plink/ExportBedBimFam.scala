package is.hail.io.plink

import is.hail.variant._

object ExportBedBimFam {

  val gtMap = Array(1, 3, 2, 0)

  def makeBedRow(gs: Iterable[Genotype], n: Int): Array[Byte] = {
    val gts = gs.hardCallIterator

    val nBytes = (n + 3) / 4
    val a = Array.ofDim[Byte](nBytes)
    var b = 0
    var k = 0
    while (k < n) {
      b |= gtMap(gts.next() + 1) << ((k & 3) * 2)
      if ((k & 3) == 3) {
        a(k >> 2) = b.toByte
        b = 0
      }
      k += 1
    }
    if ((k & 3) > 0)
      a(nBytes - 1) = b.toByte

    a
  }

  def makeBimRow(v: Variant): String = {
    val id = s"${v.contig}:${v.start}:${v.ref}:${v.alt}"
    s"""${v.contig}\t$id\t0\t${v.start}\t${v.alt}\t${v.ref}"""
  }
}
