package org.broadinstitute.hail.methods

import org.broadinstitute.hail.variant._

object ExportBedBimFam {

  def makeBedRow(pos: Int, gs: Iterable[Genotype], cutoff: Int): Array[Byte] = {
    gs.map(g =>
      if (g.gq < cutoff)
        1
      else {
        g.call.map(_.gt) match {
          case Some(0) => 3
          case Some(1) => 2
          case Some(2) => 0
          case _ => 1
        }
      })
      .grouped(4)
      .map(_.toIndexedSeq)
      .map(i =>

        if (i.size == 4) {
          i(0) | (i(1) << 2) | (i(2) << 4) | (i(3) << 6)

        }
        else {
          var ret = 0
          for ((gt, ind) <- i.zipWithIndex) {
            ret = ret | (gt << 2 * ind)
          }
          ret
        })
      .map(_.toByte)
      .toArray
  }

  def makeBimRow(v: Variant): String = {
    val id = s"${v.contig}:${v.start}:${v.ref}:${v.alt}"
    s"""${v.contig}\t$id\t0\t${v.start}\t${v.alt}\t${v.ref}"""
  }

  def makeFamRow(s: String): String = {
    s"""0 $s 0 0 0 -9"""
  }
}
