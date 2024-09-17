package is.hail.io

import is.hail.io.fs.FS
import is.hail.io.reference.LiftOver.MinMatchDefault
import is.hail.utils.{fatal, toRichIterable, using, Interval}
import is.hail.variant.{Locus, ReferenceGenome}

import scala.collection.convert.ImplicitConversions.{`collection AsScalaIterable`, `map AsScala`}
import scala.jdk.CollectionConverters.iterableAsScalaIterableConverter

import htsjdk.samtools.reference.FastaSequenceIndexEntry

package reference {

  /* ASSUMPTION: The following will not move or change for the entire duration of a hail pipeline:
   * - chainFile
   * - fastaFile
   * - indexFile */

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

    def queryInterval(interval: Interval, minMatch: Double = MinMatchDefault)
      : (Interval, Boolean) = {
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
      asJava.getContigMap.foreach { case (srcContig, destContigs) =>
        srcRG.checkContig(srcContig)
        destContigs.foreach(destRG.checkContig)
      }

    def restore(fs: FS): Unit = {
      if (!fs.isFile(chainFile))
        fatal(s"Chain file '$chainFile' does not exist, is not a file, or you do not have access.")

      using(fs.open(chainFile)) { is =>
        asJava = new htsjdk.samtools.liftover.LiftOver(is, chainFile)
      }
    }
  }

  object IndexedFastaSequenceFile {

    def apply(fs: FS, fastaFile: String, indexFile: String): IndexedFastaSequenceFile = {
      if (!fs.isFile(fastaFile))
        fatal(s"FASTA file '$fastaFile' does not exist, is not a file, or you do not have access.")

      new IndexedFastaSequenceFile(fastaFile, FastaSequenceIndex(fs, indexFile))
    }

  }

  class IndexedFastaSequenceFile private (val path: String, val index: FastaSequenceIndex)
      extends Serializable {

    def raiseIfIncompatible(rg: ReferenceGenome): Unit = {
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

    def restore(fs: FS): Unit =
      index.restore(fs)
  }

  object FastaSequenceIndex {
    def apply(fs: FS, indexFile: String): FastaSequenceIndex = {
      val index = new FastaSequenceIndex(indexFile)
      index.restore(fs)
      index
    }
  }

  class FastaSequenceIndex private (val path: String)
      extends Iterable[FastaSequenceIndexEntry] with Serializable {

    @transient var asJava: htsjdk.samtools.reference.FastaSequenceIndex = _

    def restore(fs: FS): Unit = {
      if (!fs.isFile(path))
        fatal(
          s"FASTA index file '$path' does not exist, is not a file, or you do not have access."
        )

      using(fs.open(path))(is => asJava = new htsjdk.samtools.reference.FastaSequenceIndex(is))
    }

    override def iterator: Iterator[FastaSequenceIndexEntry] =
      asJava.asScala.iterator
  }

}
