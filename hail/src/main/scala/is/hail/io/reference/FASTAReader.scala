package is.hail.io.reference

import is.hail.backend.{BroadcastValue, ExecuteContext}
import is.hail.io.fs.FS
import is.hail.utils._
import is.hail.variant.{Locus, ReferenceGenome}

import java.util
import java.util.Map.Entry
import java.util.concurrent.locks.{Lock, ReentrantLock}
import scala.collection.concurrent
import scala.language.postfixOps

import htsjdk.samtools.reference.{ReferenceSequenceFile, ReferenceSequenceFileFactory}

case class FASTAReaderConfig(
  tmpdir: String,
  fs: FS,
  rg: ReferenceGenome,
  fastaFile: String,
  indexFile: String,
  blockSize: Int = 4096,
  capacity: Int = 100,
) {
  if (blockSize <= 0)
    fatal(s"'blockSize' must be greater than 0. Found $blockSize.")

  if (capacity <= 0)
    fatal(s"'capacity' must be greater than 0. Found $capacity.")

  def reader: FASTAReader = new FASTAReader(this)
}

object FASTAReader {
  private[this] val localFastaFiles: concurrent.Map[String, String] = new concurrent.TrieMap()
  private[this] val localFastaLock: Lock = new ReentrantLock()

  def getLocalFastaFile(tmpdir: String, fs: FS, fastaFile: String, indexFile: String): String = {
    localFastaLock.lock()
    try
      localFastaFiles.getOrElseUpdate(
        fastaFile,
        FASTAReader.setup(tmpdir, fs, fastaFile, indexFile),
      )
    finally
      localFastaLock.unlock()
  }

  def setup(tmpdir: String, fs: FS, fastaFile: String, indexFile: String): String = {
    val localFastaFile = if (fastaFile.startsWith("/")) {
      fastaFile
    } else {
      val localPath = ExecuteContext.createTmpPathNoCleanup(tmpdir, "fasta-reader", "fasta")
      log.info(s"copying FASTA file at $fastaFile to $localPath")
      fs.copyRecode(fastaFile, localPath)
      localPath
    }

    val localIndexFile = localFastaFile + ".fai"
    if (localIndexFile != indexFile) {
      fs.copyRecode(indexFile, localIndexFile)
    }

    if (!fs.exists(localFastaFile))
      fatal(s"Error while copying FASTA file to local file system. Did not find '$localFastaFile'.")
    if (!fs.exists(localIndexFile))
      fatal(
        s"Error while copying FASTA index file to local file system. Did not find '$localIndexFile'."
      )

    localFastaFile
  }
}

class FASTAReader(val cfg: FASTAReaderConfig) {
  val FASTAReaderConfig(tmpdir, fs, rg, fastaFile, indexFile, blockSize, capacity) = cfg

  private[this] def newReader(): ReferenceSequenceFile = {
    val localFastaFile = FASTAReader.getLocalFastaFile(tmpdir, fs, fastaFile, indexFile)
    ReferenceSequenceFileFactory.getReferenceSequenceFile(new java.io.File(uriPath(localFastaFile)))
  }

  private[this] var reader: ReferenceSequenceFile = newReader()

  @transient private[this] lazy val cache =
    new util.LinkedHashMap[Int, String](capacity, 0.75f, true) {
      override def removeEldestEntry(eldest: Entry[Int, String]): Boolean = size() > capacity
    }

  private def hash(pos: Long): Int = (pos / blockSize).toInt

  private def getSequence(contig: String, start: Int, end: Int): String = {
    val maxEnd = rg.contigLength(contig)
    try
      reader.getSubsequenceAt(contig, start, if (end > maxEnd) maxEnd else end).getBaseString
    catch {
      // One retry, to refresh the file
      case e: htsjdk.samtools.SAMException =>
        reader = newReader()
        reader.getSubsequenceAt(contig, start, if (end > maxEnd) maxEnd else end).getBaseString
    }
  }

  private def fillBlock(blockIdx: Int): String = {
    val seq = new StringBuilder
    val start = blockIdx.toLong * blockSize
    var pos = start

    while (pos < start + blockSize && pos < rg.nBases) {
      val l = rg.globalPosToLocus(pos)
      val query = getSequence(l.contig, l.position, l.position + blockSize)
      seq ++= query
      pos += query.length
    }

    val res = seq.result()
    cache.put(blockIdx, res)
    res
  }

  private def readBlock(blockIdx: Int): String = {
    val x = cache.get(blockIdx)
    if (x != null)
      x
    else
      fillBlock(blockIdx)
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
