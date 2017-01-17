package is.hail.io.plink

import is.hail.variant._

import scala.collection.mutable

object ExportBedBimFam {

  def makeBedRow(gs: Iterable[Genotype]): Array[Byte] = {
    val ab = new mutable.ArrayBuilder.ofByte()
    var j = 0
    var b = 0
    for (g <- gs) {
      val i = g.gt match {
        case Some(0) => 3
        case Some(1) => 2
        case Some(2) => 0
        case _ => 1
      }
      b |= i << (j * 2)
      if (j == 3) {
        ab += b.toByte
        b = 0
        j = 0
      } else
        j += 1
    }
    if (j > 0)
      ab += b.toByte

    ab.result()
  }

  def makeBimRow(v: Variant): String = {
    val id = s"${v.contig}:${v.start}:${v.ref}:${v.alt}"
    s"""${v.contig}\t$id\t0\t${v.start}\t${v.alt}\t${v.ref}"""
  }
}
