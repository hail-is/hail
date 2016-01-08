package org.broadinstitute.hail.methods

import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._

object ExportBedBimFam {

  def makeBedRow(gs: Iterable[Genotype]): String = {
    val bytes = gs.map(g => g.call.map(_.gt) match {
      case Some(0) => 0
      case Some(1) => 1
      case Some(2) => 3
      case None => 2
    })
    .grouped(4)
    .map(_.toIndexedSeq)
    .map(i =>
      if (i.size == 4) {
        (i(0) << 6) & (i(1) << 4) & (i(2) << 2) & i(3)
      }
      else {
        var ret = 0
        for ((gt, ind) <- i.zipWithIndex) {
          ret = ret & (gt << (3-ind))
        }
        ret
      })
    .map(_.toByte)
    .toArray
    println(bytes.length)

    new String(bytes)
  }

  def makeBimRow(v: Variant, va: AnnotationData): String = {
    val rsid = va.vals.getOrElse("rsid", ".")
    s"""${v.contig} $rsid 0 ${v.start} ${v.ref} ${v.alt}"""
  }

  def makeFamRow(s: String): String = {
    s"""0 $s 0 0 0 0"""
  }
}
