package is.hail.io.reference

import is.hail.expr.ir.ExecuteContext
import is.hail.variant.{Locus, ReferenceGenome}
import is.hail.utils._
import is.hail.io.fs.FS

import scala.collection.JavaConverters._
import scala.collection.concurrent
import scala.language.implicitConversions

object LiftOver {
  private[this] val localChainFiles: concurrent.Map[String, String] = new concurrent.TrieMap()

  def setup(tmpdir: String, fs: FS, chainFile: String): String = {
    val localChainFile = ExecuteContext.createTmpPathNoCleanup(tmpdir, "lift-over", "chain")
    fs.copyRecode(chainFile, localChainFile)

    if (!fs.exists(localChainFile))
      fatal(s"Error while copying chain file to local file system. Did not find '$localChainFile'.")

    localChainFile
  }

  def apply(tmpdir: String, fs: FS, chainFile: String): LiftOver =
    new LiftOver(localChainFiles.getOrElseUpdate(chainFile, LiftOver.setup(tmpdir, fs, chainFile)), chainFile)
}

class LiftOver(localChainFile: String, val chainFile: String) {
  val lo = new htsjdk.samtools.liftover.LiftOver(new java.io.File(uriPath(localChainFile)))

  def queryInterval(interval: is.hail.utils.Interval, minMatch: Double = htsjdk.samtools.liftover.LiftOver.DEFAULT_LIFTOVER_MINMATCH): (is.hail.utils.Interval, Boolean) = {
    val start = interval.start.asInstanceOf[Locus]
    val end = interval.end.asInstanceOf[Locus]

    if (start.contig != end.contig)
      fatal(s"'start' and 'end' contigs must be identical. Found '$interval'.")

    val contig = start.contig
    val startPos = if (interval.includesStart) start.position else start.position + 1
    val endPos = if (interval.includesEnd) end.position else end.position - 1

    if (startPos == endPos)
      fatal(s"Cannot liftover a 0-length interval: ${ interval.toString }.\nDid you mean to use 'liftover_locus'?")

    val result = lo.liftOver(new htsjdk.samtools.util.Interval(contig, startPos, endPos), minMatch)
    if (result != null)
      (Interval(
        Locus(result.getContig, result.getStart),
        Locus(result.getContig, result.getEnd),
        includesStart = true,
        includesEnd = true),
      result.isNegativeStrand)
    else
      null
  }

  def queryLocus(l: Locus, minMatch: Double = htsjdk.samtools.liftover.LiftOver.DEFAULT_LIFTOVER_MINMATCH): (Locus, Boolean) = {
    val result = lo.liftOver(new htsjdk.samtools.util.Interval(l.contig, l.position, l.position), minMatch)
    if (result != null)
      (Locus(result.getContig, result.getStart), result.isNegativeStrand)
    else
      null
  }

  def checkChainFile(srcRG: ReferenceGenome, destRG: ReferenceGenome) {
    val cMap = lo.getContigMap.asScala
    cMap.foreach { case (srcContig, destContigs) =>
      srcRG.checkContig(srcContig)
      destContigs.asScala.foreach(destRG.checkContig)
    }
  }
}
