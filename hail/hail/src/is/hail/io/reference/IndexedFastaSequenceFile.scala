package is.hail.io.reference

import is.hail.collection.implicits.toRichIterable
import is.hail.io.fs.FS
import is.hail.utils.fatal
import is.hail.variant.ReferenceGenome

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
