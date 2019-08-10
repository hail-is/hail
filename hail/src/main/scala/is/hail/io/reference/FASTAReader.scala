package is.hail.io.reference

import java.util
import java.util.Map.Entry

import htsjdk.samtools.reference.ReferenceSequenceFileFactory
import is.hail.HailContext
import is.hail.utils._
import is.hail.variant.{Locus, ReferenceGenome}
import org.apache.commons.io.IOUtils
import is.hail.io.fs.FS

import scala.language.postfixOps
import scala.collection.concurrent

class SerializableReferenceSequenceFile(val fs: FS, val fastaFile: String, val indexFile: String) extends Serializable {
  @transient lazy val value = {
    val localFastaFile = FASTAReader.getLocalFastaFileName(fs, fastaFile, indexFile)
    ReferenceSequenceFileFactory.getReferenceSequenceFile(new java.io.File(localFastaFile))
  }
}

object FASTAReader {
  private[this] val localFastaFiles: concurrent.Map[String, String] = new concurrent.TrieMap()

  def getLocalIndexFileName(fastaFile: String): String =
    ReferenceSequenceFileFactory.getFastaIndexFileName(new java.io.File(fastaFile).toPath).toString

  def getLocalFastaFileName(fs: FS, fastaFile: String, indexFile: String): String =
    localFastaFiles.getOrElseUpdate(fastaFile, FASTAReader.setup(fs, fastaFile, indexFile))

  def getUriLocalIndexFile(fs: FS, indexFile: String): String = {
    val tmpDir = TempDir(fs)
    val localIndexFile = tmpDir.createLocalTempFile(extension = "fai")
    fs.copy(indexFile, localIndexFile)
    uriPath(localIndexFile)
  }

  def setup(fs: FS, fastaFile: String, indexFile: String): String = {
    val tmpDir = TempDir(fs)
    val localFastaFile = tmpDir.createLocalTempFile(extension = "fasta")
    val uriLocalFastaFile = uriPath(localFastaFile)

    fs.readFile(fastaFile) { in =>
      fs.writeFile(localFastaFile) { out =>
        IOUtils.copy(in, out)
      }}

    val localIndexFile = "file://" + getLocalIndexFileName(uriLocalFastaFile)
    fs.copy(indexFile, localIndexFile)

    if (!fs.exists(localFastaFile))
      fatal(s"Error while copying FASTA file to local file system. Did not find '$localFastaFile'.")
    if (!fs.exists(localIndexFile))
      fatal(s"Error while copying FASTA index file to local file system. Did not find '$localIndexFile'.")

    uriLocalFastaFile
  }

  def apply(hc: HailContext, rg: ReferenceGenome, fastaFile: String, indexFile: String,
    blockSize: Int = 4096, capacity: Int = 100): FASTAReader = {
    if (blockSize <= 0)
      fatal(s"'blockSize' must be greater than 0. Found $blockSize.")
    if (capacity <= 0)
      fatal(s"'capacity' must be greater than 0. Found $capacity.")

    new FASTAReader(hc.sFS, rg, fastaFile, indexFile, blockSize, capacity)
  }
}

class FASTAReader(val fs: FS, val rg: ReferenceGenome,
  val fastaFile: String, val indexFile: String, val blockSize: Int, val capacity: Int) extends Serializable {

  val reader = new SerializableReferenceSequenceFile(fs, fastaFile, indexFile)
  assert(reader.value.isIndexed)

  @transient private[this] lazy val cache = new util.LinkedHashMap[Int, String](capacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[Int, String]): Boolean = size() > capacity
  }

  private def hash(pos: Long): Int = (pos / blockSize).toInt

  private def getSequence(contig: String, start: Int, end: Int): String = {
    val maxEnd = rg.contigLength(contig)
    reader.value.getSubsequenceAt(contig, start, if (end > maxEnd) maxEnd else end).getBaseString
  }

  private def fillBlock(blockIdx: Int) {
    val seq = new StringBuilder
    val start = blockIdx.toLong * blockSize
    var pos = start

    while (pos < start + blockSize && pos < rg.nBases) {
      val l = rg.globalPosToLocus(pos)
      val query = getSequence(l.contig, l.position, l.position + blockSize)
      seq ++= query
      pos += query.length
    }

    cache.put(blockIdx, seq.result())
  }

  private def readBlock(blockIdx: Int): String = {
    if (!cache.containsKey(blockIdx))
      fillBlock(blockIdx)
    cache.get(blockIdx)
  }

  private def readBlock(blockIdx: Int, offset: Int): String = {
    require(offset >= 0 && offset < blockSize)
    readBlock(blockIdx)(offset).toString
  }

  private def readBlock(blockIdx: Int, offset: Int, size: Int): String = {
    require(offset >= 0 && offset + size <= blockSize)
    readBlock(blockIdx).substring(offset, offset + size)
  }

  def lookup(contig: String, pos: Int, before: Int, after: Int): String = {
    assert(rg.isValidLocus(contig, pos))
    if (before == 0 && after == 0)
      lookupGlobalPos(rg.locusToGlobalPos(contig, pos))
    else {
      val start = rg.locusToGlobalPos(contig, math.max(pos - before, 1))
      val end = rg.locusToGlobalPos(contig, math.min(pos + after, rg.contigLength(contig)))
      lookupGlobalInterval(start, end)
    }
  }

  def lookup(interval: Interval): String = {
    val start = interval.start.asInstanceOf[Locus]
    val end = interval.end.asInstanceOf[Locus]
    assert(rg.isValidLocus(start) && rg.isValidLocus(end))

    var startGlobalPos = rg.locusToGlobalPos(start)
    var endGlobalPos = rg.locusToGlobalPos(end)

    if (!interval.includesStart)
      startGlobalPos += 1
    if (!interval.includesEnd)
      endGlobalPos -= 1

    assert(startGlobalPos >= 0 && endGlobalPos < rg.nBases)

    lookupGlobalInterval(startGlobalPos, endGlobalPos)
  }

  private def lookupGlobalPos(pos: Long): String = {
    val blockIdx = hash(pos)
    readBlock(blockIdx, (pos % blockSize).toInt)
  }

  private def lookupGlobalInterval(start: Long, end: Long): String = {
    require(end >= start)
    val seq = new StringBuilder
    var pos = start

    while (pos <= end) {
      val blockIdx = hash(pos)
      val offset = (pos % blockSize).toInt
      val maxSize = blockSize - offset
      val nRemaining = end - pos + 1
      val size = if (nRemaining > maxSize) maxSize else nRemaining
      seq ++= readBlock(blockIdx, offset, size.toInt)
      pos += size
    }

    seq.result()
  }
}
