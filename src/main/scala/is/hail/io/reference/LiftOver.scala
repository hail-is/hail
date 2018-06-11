package is.hail.io.reference

import is.hail.HailContext
import is.hail.variant.{Locus, ReferenceGenome}
import is.hail.utils._
import org.apache.commons.io.IOUtils
import org.apache.hadoop.conf.Configuration

import scala.collection.JavaConverters._
import scala.collection.concurrent
import scala.language.implicitConversions

class SerializableHtsjdkLiftOver(val hConf: SerializableHadoopConfiguration, val chainFile: String) extends Serializable {
  @transient lazy val value = {
    val localChainFile = LiftOver.getLocalChainFileName(hConf.value, chainFile)
    new htsjdk.samtools.liftover.LiftOver(new java.io.File(localChainFile))
  }
}

object LiftOver {
  private[this] val localChainFiles: concurrent.Map[String, String] = new concurrent.TrieMap()

  def getLocalChainFileName(hConf: Configuration, chainFile: String): String =
    localChainFiles.getOrElseUpdate(chainFile, LiftOver.setup(hConf, chainFile))

  def setup(hConf: Configuration, chainFile: String): String = {
    val tmpDir = TempDir(hConf)
    val localChainFile = tmpDir.createLocalTempFile(extension = "chain")

    hConf.readFile(chainFile) { in =>
      hConf.writeFile(localChainFile) { out =>
        IOUtils.copy(in, out)
      }}

    if (!hConf.exists(localChainFile))
      fatal(s"Error while copying chain file to local file system. Did not find '$localChainFile'.")

    uriPath(localChainFile)
  }

  def apply(hc: HailContext, chainFile: String): LiftOver = {
    val hConf = new SerializableHadoopConfiguration(hc.hadoopConf)
    new LiftOver(hConf, chainFile)
  }
}

class LiftOver(val hConf: SerializableHadoopConfiguration, val chainFile: String) extends Serializable {
  val lo = new SerializableHtsjdkLiftOver(hConf, chainFile)

  def queryInterval(interval: is.hail.utils.Interval, minMatch: Double = htsjdk.samtools.liftover.LiftOver.DEFAULT_LIFTOVER_MINMATCH): is.hail.utils.Interval = {
    val start = interval.start.asInstanceOf[Locus]
    val end = interval.end.asInstanceOf[Locus]

    if (start.contig != end.contig)
      fatal(s"'start' and 'end' contigs must be identical. Found '$interval'.")

    val contig = start.contig
    val startPos = if (interval.includesStart) start.position else start.position + 1
    val endPos = if (interval.includesEnd) end.position else end.position - 1

    val result = lo.value.liftOver(new htsjdk.samtools.util.Interval(contig, startPos, endPos), minMatch)
    if (result != null)
      Interval(
        Locus(result.getContig, result.getStart),
        Locus(result.getContig, result.getEnd),
        includesStart = true,
        includesEnd = true)
    else
      null
  }

  def queryLocus(l: Locus, minMatch: Double = htsjdk.samtools.liftover.LiftOver.DEFAULT_LIFTOVER_MINMATCH): Locus = {
    val result = lo.value.liftOver(new htsjdk.samtools.util.Interval(l.contig, l.position, l.position), minMatch)
    if (result != null)
      Locus(result.getContig, result.getStart)
    else
      null
  }

  def checkChainFile(srcRG: ReferenceGenome, destRG: ReferenceGenome) {
    val cMap = lo.value.getContigMap.asScala
    cMap.foreach { case (srcContig, destContigs) =>
      srcRG.checkContig(srcContig)
      destContigs.asScala.foreach(destRG.checkContig)
    }
  }
}
