package is.hail.io.reference

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.util
import java.util.Map.Entry

import htsjdk.samtools.reference.{FastaSequenceIndex, FastaSequenceIndexCreator, ReferenceSequenceFile, ReferenceSequenceFileFactory}
import is.hail.HailContext
import is.hail.utils._
import is.hail.variant.{RGBase, ReferenceGenome, Locus}
import org.apache.hadoop

import scala.language.postfixOps
import scala.sys.process._

class SerializableReferenceSequenceFile(val hConf: SerializableHadoopConfiguration, val fastaFile: String, val indexFile: String) extends Serializable {
  @transient var value_ : ReferenceSequenceFile = _

  def value: ReferenceSequenceFile = {
    if (value_ == null) {
      val localFastaFile = FASTAReader.getLocalFastaFileName(hConf, fastaFile, indexFile)
      value_ = ReferenceSequenceFileFactory.getReferenceSequenceFile(new java.io.File(localFastaFile))
    }
    value_
  }
}

object FASTAReader {
  @volatile var localFastaFiles = Map.empty[String, String]

  def getLocalIndexFileName(fastaFile: String): String =
    ReferenceSequenceFileFactory.getFastaIndexFileName(new java.io.File(fastaFile).toPath).toString

  def getLocalFastaFileName(hConf: SerializableHadoopConfiguration, fastaFile: String, indexFile: String): String = {
    synchronized {
      if (!localFastaFiles.contains(fastaFile)) {
        val localFile = FASTAReader.setup(hConf, fastaFile, indexFile)
        localFastaFiles += fastaFile -> localFile
      }
      localFastaFiles(fastaFile)
    }
  }

  def setup(hConf: SerializableHadoopConfiguration, fastaFile: String, indexFile: String): String = {
    val fastaPath = new hadoop.fs.Path(fastaFile)
    var localFastaFile = hConf.value.getTemporaryFile("file:///tmp/", suffix = Some(fastaPath.getName))
    var uriLocalFastaFile = uriPath(localFastaFile)

    hConf.value.copy(fastaFile, localFastaFile)

    val isCompressed = Set(".gz", ".bgz").contains(hConf.value.getCodec(fastaFile))
    if (isCompressed) {
      val ec = s"gunzip -f $uriLocalFastaFile" !

      if (ec != 0)
        fatal(s"Error decompressing FASTA file: `gunzip -f $uriLocalFastaFile'")

      uriLocalFastaFile = hConf.value.stripCodec(uriLocalFastaFile)
      localFastaFile = hConf.value.stripCodec(localFastaFile)
    }

    val localIndexFile = "file://" + getLocalIndexFileName(uriLocalFastaFile)
    hConf.value.copy(indexFile, localIndexFile)

    if (!hConf.value.exists(localFastaFile))
      fatal(s"Error while copying FASTA file to local file system. Did not find '$localFastaFile'.")
    if (!hConf.value.exists(localIndexFile))
      fatal(s"Error while copying FASTA index file to local file system. Did not find '$localIndexFile'.")

    uriLocalFastaFile
  }

  def apply(hc: HailContext, rg: ReferenceGenome, fastaFile: String, indexFile: String,
    blockSize: Int = 4096, capacity: Int = 100): FASTAReader = {
    val hConf = new SerializableHadoopConfiguration(hc.hadoopConf)

    if (blockSize <= 0)
      fatal(s"'blockSize' must be greater than 0. Found $blockSize.")
    if (capacity <= 0)
      fatal(s"'capacity' must be greater than 0. Found $capacity.")

    new FASTAReader(hConf, rg, fastaFile, indexFile, blockSize, capacity)
  }
}

class FASTAReader(hConf: SerializableHadoopConfiguration, rg: ReferenceGenome,
  fastaFile: String, indexFile: String, blockSize: Int, capacity: Int) extends Serializable {

  val reader = new SerializableReferenceSequenceFile(hConf, fastaFile, indexFile)
  assert(reader.value.isIndexed)

  @transient private lazy val cache = new util.LinkedHashMap[Int, String](capacity, 0.75f, true) {
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

    if (!interval.includeStart)
      startGlobalPos += 1
    if (!interval.includeEnd)
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
