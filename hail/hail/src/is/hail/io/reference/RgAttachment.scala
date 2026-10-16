package is.hail.io.reference

import is.hail.collection.implicits.toRichIterable
import is.hail.io.fs.FS
import is.hail.io.reference.Liftover.MinMatchDefault
import is.hail.utils.{fatal, using, Interval}
import is.hail.variant.{Locus, ReferenceGenome}

import scala.jdk.CollectionConverters._

/** A liftover or reference sequence held by a session, attached to the reference genomes it holds.
  * It is never part of those reference genomes, and defines no equality: two attachments are the
  * same only if they are the same object.
  */
sealed trait RgAttachment extends Serializable {

  /** Readies this attachment, which may have been deserialized, for use through `fs`. Files that
    * must be local are copied under `localTmpdir`.
    */
  def restore(fs: FS, localTmpdir: String): Unit
}

object Liftover {

  /** Loads the liftover from `source` to `dest` defined by `chainFile`, checking that the chain
    * file's contigs exist in both reference genomes.
    */
  def apply(fs: FS, source: ReferenceGenome, dest: ReferenceGenome, chainFile: String): Liftover = {
    if (source.name == dest.name)
      fatal(
        s"Destination reference genome cannot have the same name as this reference '${source.name}'"
      )

    val lo = new Liftover(source, dest, chainFile)
    lo.loadChain(fs)
    lo.checkChainFile()
    lo
  }

  val MinMatchDefault: Double = htsjdk.samtools.liftover.LiftOver.DEFAULT_LIFTOVER_MINMATCH
}

/** A mapping of loci from `source` to `dest`, defined by `chainFile`, whose contigs were checked
  * against exactly these reference genomes. The chain is not serialized: [[restore]] reloads it.
  */
final class Liftover private (
  val source: ReferenceGenome,
  val dest: ReferenceGenome,
  val chainFile: String,
) extends RgAttachment {

  def key: (ReferenceGenome, ReferenceGenome) = (source, dest)

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

  private def checkChainFile(): Unit =
    asJava.getContigMap.asScala.foreach { case (srcContig, destContigs) =>
      source.checkContig(srcContig)
      destContigs.asScala.foreach(dest.checkContig)
    }

  /** Loads the chain through `fs`. A liftover has no local files, so `localTmpdir` is unused. */
  override def restore(fs: FS, localTmpdir: String): Unit =
    loadChain(fs)

  private def loadChain(fs: FS): Unit = synchronized {
    if (asJava == null) {
      if (!fs.isFile(chainFile))
        fatal(s"Chain file '$chainFile' does not exist, is not a file, or you do not have access.")

      using(fs.open(chainFile)) { is =>
        asJava = new htsjdk.samtools.liftover.LiftOver(is, chainFile)
      }
    }
  }
}

object ReferenceSequence {

  /** The reference sequence of `rg` given by `fastaFile` and its index `indexFile`, checking that
    * the index's contigs and lengths match `rg`.
    */
  def apply(fs: FS, rg: ReferenceGenome, fastaFile: String, indexFile: String)
    : ReferenceSequence = {
    if (!fs.isFile(fastaFile))
      fatal(s"FASTA file '$fastaFile' does not exist, is not a file, or you do not have access.")

    raiseIfIncompatible(rg, fastaFile, FastaSequenceIndex(fs, indexFile))
    new ReferenceSequence(rg, fastaFile, indexFile)
  }

  private def raiseIfIncompatible(rg: ReferenceGenome, path: String, index: FastaSequenceIndex)
    : Unit = {
    val jindex = index.asJava

    val missingContigs = rg.contigs.filterNot(jindex.hasIndexEntry)
    if (missingContigs.nonEmpty)
      fatal(
        s"Contigs missing in FASTA '$path' that are present in reference genome '${rg.name}':\n  " +
          s"@1",
        missingContigs.truncatable("\n  "),
      )

    val invalidLengths =
      for {
        (contig, length) <- rg.lengths
        fastaLength = jindex.getIndexEntry(contig).getSize
        if fastaLength != length
      } yield (contig, length, fastaLength)

    if (invalidLengths.nonEmpty)
      fatal(
        s"Contig sizes in FASTA '$path' do not match expected sizes for reference genome '${rg.name}':\n  " +
          s"@1",
        invalidLengths.map { case (c, e, f) => s"$c\texpected:$e\tfound:$f" }.truncatable("\n  "),
      )
  }
}

/** The bases of `rg`, given by the FASTA file `fasta` and its index `index`. Nothing is read until
  * [[restore]], which every copy, including a deserialized one, needs before a lookup.
  */
final class ReferenceSequence private (
  val rg: ReferenceGenome,
  val fasta: String,
  val index: String,
) extends RgAttachment {

  @transient @volatile private[this] var readerCfg: FASTAReaderConfig = _

  @transient private[this] lazy val readers: ThreadLocal[FASTAReader] =
    new ThreadLocal[FASTAReader]

  /** Readies lookups to read through `fs`, copying the FASTA file under `localTmpdir` if it is not
    * local. Readers made with a different `fs` or `localTmpdir` are replaced on their next lookup.
    */
  override def restore(fs: FS, localTmpdir: String): Unit =
    readerCfg = FASTAReaderConfig(
      localTmpdir,
      fs,
      rg,
      fs.parseUrl(fasta).toString,
      fs.parseUrl(index).toString,
    )

  private[this] def reader(): FASTAReader = {
    val cfg = readerCfg
    var r = readers.get()
    if (r == null || r.cfg != cfg) {
      r = cfg.reader
      readers.set(r)
    }
    r
  }

  def lookup(contig: String, position: Int, before: Int, after: Int): String =
    reader().lookup(contig, position, before, after)
}
