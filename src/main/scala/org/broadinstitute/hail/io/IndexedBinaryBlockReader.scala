package org.broadinstitute.hail.io

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._
import org.apache.hadoop.mapred.LineRecordReader
import scala.collection.mutable.ArrayBuffer

case class BlockIndex(start: Long, length: Int, variantIndex: Int)

abstract class IndexedBinaryBlockReader(job: Configuration, split: FileSplit)
  extends RecordReader[RichLongWritable, ByteBlock] {

  private val LOG: Log = LogFactory.getLog(classOf[IndexedBinaryBlockReader].getName)
  private val start: Long = split.getStart
  private var pos: Long = start
  private val end: Long = start + split.getLength
  private val file: Path = split.getPath
  private val fs: FileSystem = file.getFileSystem(job)
  private val bfis: BetterFSDataInputStream = new BetterFSDataInputStream(fs.open(file))
  private val blocksToProcess: Array[(BlockIndex)] = getIndices(start, end)
  private var index = 0


  println("nBlocks=%d, blocks(0)=%d, bytes %d:%d".format(blocksToProcess.length, blocksToProcess(0).start, start, end))

  def getIndices(start: Long, end: Long): Array[BlockIndex]

  def next(key: RichLongWritable, value: ByteBlock): Boolean = {
    if (index >= blocksToProcess.length) {
      return false
    }

    val thisBlock = blocksToProcess(index)

    if (thisBlock.length < 0) {
      return false
    }

    if (pos != thisBlock.start){
      bfis.seek(thisBlock.start)
      pos = thisBlock.start
    }

    val bytes = bfis.readBytes(thisBlock.length)
    pos += thisBlock.length

    value.setBlock(bytes)
    value.setIndex(thisBlock.variantIndex)

    index += 1
    true
  }

  def createKey(): RichLongWritable = {
    new RichLongWritable()
  }

  def createValue(): ByteBlock = {
    new ByteBlock()
  }

  def getPos: Long = pos

  def getProgress: Float = {
    if (start == end) {
      0.0f
    }
    else {
      Math.min(1.0f, (pos - start) / (end - start).toFloat)
    }
  }

  def close(): Unit = {
    bfis.close()
  }
}

class RichLongWritable extends LongWritable with Serializable
