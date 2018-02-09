package is.hail.io.reference

import java.util
import java.util.Map.Entry

import htsjdk.samtools.reference.{FastaSequenceIndex, IndexedFastaSequenceFile}
import is.hail.utils._
import is.hail.variant.{GRBase, GenomeReference, Locus}


class FastaReader(grBase: GRBase, file: String, indexFile: String,
  blockSize: Int = 4096, capacity: Int = 100) extends Serializable {
  if (blockSize <= 0)
    fatal(s"`blockSize' must be greater than 0. Found $blockSize.")
  if (capacity <= 0)
    fatal(s"`capacity' must be greater than 0. Found $capacity.")

  val gr = grBase.asInstanceOf[GenomeReference]
  val index = new FastaSequenceIndex(new java.io.File(indexFile))
  val reader = new IndexedFastaSequenceFile(new java.io.File(file), index)

  val missingContigs = gr.contigs.filterNot(index.hasIndexEntry)
  if (missingContigs.nonEmpty)
    fatal(s"Contigs missing in FASTA index file `$indexFile' that are present in reference genome `${ gr.name }':\n  " +
      s"@1", missingContigs.truncatable("\n  "))

  val invalidLengths = gr.contigs.flatMap { c =>
    val expectedSize = gr.contigLength(c)
    val indexSize = index.getIndexEntry(c).getSize
    if (indexSize != expectedSize)
      Some((c, expectedSize, indexSize))
    else
      None
  }.map { case (c, e, f) => s"$c\texpected:$e\tfound:$f"}

  if (invalidLengths.nonEmpty)
    fatal(s"Contig sizes in FASTA index file `$indexFile' do not match expected sizes for reference genome `${ gr.name }':\n  " +
      s"@1", invalidLengths.truncatable("\n  "))

  private val cache = new util.LinkedHashMap[Int, String](capacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[Int, String]): Boolean = size() > capacity
  }

  private def hash(pos: Long): Int = (pos / blockSize).toInt

  private def getSequence(contig: String, start: Int, end: Int): String = {
    val maxEnd = gr.contigLength(contig)
    reader.getSubsequenceAt(contig, start, if (end > maxEnd) maxEnd else end).getBaseString
  }

  private def fillBlock(blockIdx: Int) {
    var seq = new StringBuilder
    val startGlobalPos = blockIdx.toLong * blockSize
    var curGlobalPos = startGlobalPos

    while (curGlobalPos < startGlobalPos + blockSize && curGlobalPos < gr.nBases) {
      val l = gr.globalPosToLocus(curGlobalPos)
      val query = getSequence(l.contig, l.position, l.position + blockSize)
      seq ++= query
      curGlobalPos += query.length
    }

    cache.put(blockIdx, seq.result())
  }

  private def readBlock(blockIdx: Int, offset: Int): String = {
    require(offset >= 0 && offset < blockSize)
    if (!cache.containsKey(blockIdx))
      fillBlock(blockIdx)
    cache.get(blockIdx)(offset).toString
  }

  def lookupLocus(l: Locus): String = {
    lookupGlobalPos(gr.locusToGlobalIndex(l))
  }

  def lookupGlobalPos(pos: Long): String = {
    val blockIdx = hash(pos)
    readBlock(blockIdx, (pos % blockSize).toInt)
  }
}
