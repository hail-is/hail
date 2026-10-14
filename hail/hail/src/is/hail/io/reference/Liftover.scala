package is.hail.io.reference

import is.hail.io.fs.FS
import is.hail.io.reference.Liftover.MinMatchDefault
import is.hail.utils.{fatal, using, Interval}
import is.hail.variant.{Locus, ReferenceGenome}

import scala.collection.mutable
import scala.jdk.CollectionConverters._

object Liftover {

  /** Loads the liftover from `source` to `dest` defined by `chainFile`, checking that the chain
    * file's contigs exist in both reference genomes.
    */
  def apply(fs: FS, source: ReferenceGenome, dest: ReferenceGenome, chainFile: String): Liftover = {
    if (source.name == dest.name)
      fatal(
        s"Destination reference genome cannot have the same name as this reference '${source.name}'"
      )

    val lo = new Liftover(source.name, dest.name, chainFile)
    lo.restore(fs)
    lo.checkChainFile(source, dest)
    lo
  }

  /** Adds `lo` to a session's liftovers, which hold at most one per source and destination. */
  def addFatalOnCollision(existing: mutable.Map[(String, String), Liftover], lo: Liftover): Unit = {
    if (existing.contains(lo.key))
      fatal(
        s"LiftOver already exists for source reference '${lo.source}' and destination reference '${lo.dest}'."
      )

    existing += lo.key -> lo
  }

  val MinMatchDefault: Double = htsjdk.samtools.liftover.LiftOver.DEFAULT_LIFTOVER_MINMATCH
}

/** A mapping of loci from the reference genome named `source` to the one named `dest`, defined by
  * `chainFile`. The chain is not serialized: [[restore]] reloads it.
  */
class Liftover private (val source: String, val dest: String, val chainFile: String)
    extends Serializable {

  def key: (String, String) = (source, dest)

  @transient @volatile private[this] var asJava: htsjdk.samtools.liftover.LiftOver = _

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

  private def checkChainFile(srcRG: ReferenceGenome, destRG: ReferenceGenome): Unit =
    asJava.getContigMap.asScala.foreach { case (srcContig, destContigs) =>
      srcRG.checkContig(srcContig)
      destContigs.asScala.foreach(destRG.checkContig)
    }

  def restore(fs: FS): Unit = synchronized {
    if (asJava == null) {
      if (!fs.isFile(chainFile))
        fatal(s"Chain file '$chainFile' does not exist, is not a file, or you do not have access.")

      using(fs.open(chainFile)) { is =>
        asJava = new htsjdk.samtools.liftover.LiftOver(is, chainFile)
      }
    }
  }
}
