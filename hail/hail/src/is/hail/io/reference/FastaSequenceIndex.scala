package is.hail.io.reference

import is.hail.io.fs.FS
import is.hail.utils.{fatal, using}

import scala.jdk.CollectionConverters._

import htsjdk.samtools.reference.FastaSequenceIndexEntry

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
