package is.hail.io.plink

import is.hail.variant._

object ExportBedBimFam {

  val gtMap = Array(1, 3, 2, 0)

  def makeBedRow(gs: Iterable[Genotype], n: Int): Array[Byte] = {
    val gts = gs.hardCallIterator

    val nBytes = (n + 3) / 4
    val a = Array.ofDim[Byte](nBytes)
    var j = 0
    var b = 0
    var k = 0
    var k4 = 0
    while (k < n) {
      b |= gtMap(gts.nextInt() + 1) << (j * 2)
      if (j == 3) {
        a(k4) = b.toByte
        b = 0
        j = 0
        k4 += 1
      } else
        j += 1
      k += 1
    }
    if (j > 0)
      a(nBytes - 1) = b.toByte

    a
  }

  def makeBimRow(v: Variant): String = {
    val id = s"${v.contig}:${v.start}:${v.ref}:${v.alt}"
    s"""${v.contig}\t$id\t0\t${v.start}\t${v.alt}\t${v.ref}"""
  }
}
