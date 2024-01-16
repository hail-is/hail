package is.hail.io.reference
import is.hail.io.fs.FS
import is.hail.utils._
import is.hail.variant.{Locus, ReferenceGenome}

import scala.collection.JavaConverters._

object LiftOver {
  def apply(fs: FS, chainFile: String): LiftOver = new LiftOver(fs, chainFile)
}

class LiftOver(fs: FS, val chainFile: String) {
  val lo = using(fs.open(chainFile))(new htsjdk.samtools.liftover.LiftOver(_, chainFile))

  def queryInterval(
    interval: is.hail.utils.Interval,
    minMatch: Double = htsjdk.samtools.liftover.LiftOver.DEFAULT_LIFTOVER_MINMATCH,
  ): (is.hail.utils.Interval, Boolean) = {
    val start = interval.start.asInstanceOf[Locus]
    val end = interval.end.asInstanceOf[Locus]

    if (start.contig != end.contig)
      fatal(s"'start' and 'end' contigs must be identical. Found '$interval'.")

    val contig = start.contig
    val startPos = if (interval.includesStart) start.position else start.position + 1
    val endPos = if (interval.includesEnd) end.position else end.position - 1

    if (startPos == endPos)
      fatal(
        s"Cannot liftover a 0-length interval: ${interval.toString}.\nDid you mean to use 'liftover_locus'?"
      )

    val result = lo.liftOver(new htsjdk.samtools.util.Interval(contig, startPos, endPos), minMatch)
    if (result != null)
      (
        Interval(
          Locus(result.getContig, result.getStart),
          Locus(result.getContig, result.getEnd),
          includesStart = true,
          includesEnd = true,
        ),
        result.isNegativeStrand,
      )
    else
      null
  }

  def queryLocus(
    l: Locus,
    minMatch: Double = htsjdk.samtools.liftover.LiftOver.DEFAULT_LIFTOVER_MINMATCH,
  ): (Locus, Boolean) = {
    val result =
      lo.liftOver(new htsjdk.samtools.util.Interval(l.contig, l.position, l.position), minMatch)
    if (result != null)
      (Locus(result.getContig, result.getStart), result.isNegativeStrand)
    else
      null
  }

  def checkChainFile(srcRG: ReferenceGenome, destRG: ReferenceGenome): Unit = {
    val cMap = lo.getContigMap.asScala
    cMap.foreach { case (srcContig, destContigs) =>
      srcRG.checkContig(srcContig)
      destContigs.asScala.foreach(destRG.checkContig)
    }
  }
}
