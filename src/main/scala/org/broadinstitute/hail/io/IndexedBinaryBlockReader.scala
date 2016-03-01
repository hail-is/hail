package org.broadinstitute.hail.io

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._
import org.apache.hadoop.mapred.LineRecordReader
import org.broadinstitute.hail.variant.{Variant, Genotype}
import scala.collection.mutable.ArrayBuffer

case class BlockIndex(start: Long, length: Int, variantIndex: Int) {
  override def toString: String = s"start=$start, length=$length, index=$variantIndex"
}

abstract class ParsedLine[K] extends Serializable {
  def setGS(gs: Iterable[Genotype])
  def getGS: Iterable[Genotype]
  def setKey(k: K)
  def getKey: K
  def setLog(l: String)
  def getLog: String
}

class BgenParsedLine extends ParsedLine[Variant] {
  var gs: Iterable[Genotype] = null
  var variant: Variant = null
  var log: String = ""

  def setGS(gs: Iterable[Genotype]) {
    this.gs = gs
  }
  def getGS: Iterable[Genotype] = gs

  def setKey(k: Variant) {
    variant = k
  }
  def getKey: Variant = variant

  def setLog(l: String) {
    log = l
  }

  def getLog: String = log
}

class PlinkParsedLine extends ParsedLine[Int] {
  var gs: Iterable[Genotype] = null
  var pos: Int = -1
  var log: String = ""

  def setGS(gs: Iterable[Genotype]) {
    this.gs = gs
  }
  def getGS: Iterable[Genotype] = gs

  def setKey(k: Int) {
    pos = k
  }
  def getKey: Int = pos

  def setLog(l: String) {
    log = l
  }

  def getLog: String = log
}

abstract class IndexedBinaryBlockReader[K](job: Configuration, split: FileSplit)
  extends RecordReader[LongWritable, ParsedLine[K]] {

  val LOG: Log = LogFactory.getLog(classOf[IndexedBinaryBlockReader[K]].getName)
  val partitionStart: Long = split.getStart
  var pos: Long = partitionStart
  val end: Long = partitionStart + split.getLength - 1
  val bfis = openFile
  //println(s"path=${split.getPath}, start=${split.getStart}, length=${split.getLength}")

  def openFile: HadoopFSDataBinaryReader = {
    val file: Path = split.getPath
    val fs: FileSystem = file.getFileSystem(job)
    new HadoopFSDataBinaryReader(fs.open(file))
  }

  def seekToFirstBlock(start: Long): Unit

  def next(key: LongWritable, value: ParsedLine[K]): Boolean

  def createKey(): LongWritable = new LongWritable()

  def createValue(): ParsedLine[K]

  def getPos: Long = pos

  def getProgress: Float = {
    if (partitionStart == end) {
      0.0f
    }
    else {
      Math.min(1.0f, (pos - partitionStart) / (end - partitionStart).toFloat)
    }
  }

  def close() = bfis.close()

}