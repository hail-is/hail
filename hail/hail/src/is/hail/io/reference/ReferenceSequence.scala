package is.hail.io.reference

import is.hail.collection.implicits.toRichIterable
import is.hail.io.fs.FS
import is.hail.utils.fatal
import is.hail.variant.ReferenceGenome

import scala.collection.mutable

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

  /** Adds `seq` to a session's reference sequences, which hold at most one per reference genome. */
  def addFatalOnCollision(existing: mutable.Map[String, ReferenceSequence], seq: ReferenceSequence)
    : Unit = {
    if (existing.contains(seq.rg.name))
      fatal(s"FASTA sequence has already been loaded for reference genome '${seq.rg.name}'.")

    existing += seq.rg.name -> seq
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
class ReferenceSequence private (val rg: ReferenceGenome, val fasta: String, val index: String)
    extends Serializable {

  @transient @volatile private[this] var readerCfg: FASTAReaderConfig = _

  @transient private[this] lazy val readers: ThreadLocal[FASTAReader] =
    new ThreadLocal[FASTAReader]

  /** Readies lookups to read through `fs`, copying the FASTA file under `tmpdir` if it is not
    * local. Readers made with a different `fs` or `tmpdir` are replaced on their next lookup.
    */
  def restore(tmpdir: String, fs: FS): Unit =
    readerCfg =
      FASTAReaderConfig(tmpdir, fs, rg, fs.parseUrl(fasta).toString, fs.parseUrl(index).toString)

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
