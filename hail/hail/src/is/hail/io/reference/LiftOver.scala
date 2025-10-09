package is.hail.io.reference

import is.hail.io.fs.FS
import is.hail.io.reference.LiftOver.MinMatchDefault
import is.hail.utils.{fatal, using, Interval}
import is.hail.variant.{Locus, ReferenceGenome}

import scala.jdk.CollectionConverters._

object LiftOver {
  def apply(fs: FS, chainFile: String): LiftOver = {
    val lo = new LiftOver(chainFile)
    lo.restore(fs)
    lo
  }

  val MinMatchDefault: Double = htsjdk.samtools.liftover.LiftOver.DEFAULT_LIFTOVER_MINMATCH
}

class LiftOver private (chainFile: String) extends Serializable {

  @transient var asJava: htsjdk.samtools.liftover.LiftOver = _

  def queryInterval(interval: Interval, minMatch: Double = MinMatchDefault): (Interval, Boolean) = {
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

    val result = asJava.liftOver(
      new htsjdk.samtools.util.Interval(contig, startPos, endPos),
      minMatch,
    )
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

  def queryLocus(l: Locus, minMatch: Double = MinMatchDefault): (Locus, Boolean) = {
    val result = asJava.liftOver(
      new htsjdk.samtools.util.Interval(l.contig, l.position, l.position),
      minMatch,
    )
    if (result != null) (Locus(result.getContig, result.getStart), result.isNegativeStrand)
    else null
  }

  def checkChainFile(srcRG: ReferenceGenome, destRG: ReferenceGenome): Unit =
    asJava.getContigMap.asScala.foreach { case (srcContig, destContigs) =>
      srcRG.checkContig(srcContig)
      destContigs.asScala.foreach(destRG.checkContig)
    }

  def restore(fs: FS): Unit = {
    if (!fs.isFile(chainFile))
      fatal(s"Chain file '$chainFile' does not exist, is not a file, or you do not have access.")

    using(fs.open(chainFile)) { is =>
      asJava = new htsjdk.samtools.liftover.LiftOver(is, chainFile)
    }
  }
}
