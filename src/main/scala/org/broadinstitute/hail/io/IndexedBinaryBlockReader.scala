package org.broadinstitute.hail.io

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._
import org.apache.hadoop.mapred.LineRecordReader
import org.broadinstitute.hail.variant.{Variant, Genotype}
import scala.collection.mutable.ArrayBuffer
import org.broadinstitute.hail.annotations._

/*case class BlockIndex(start: Long, length: Int, variantIndex: Int) {
  override def toString: String = s"start=$start, length=$length, index=$variantIndex"
}*/

abstract class ParsedLine[K] extends Serializable {
  def setGS(gs: Iterable[Genotype])
  def getGS: Iterable[Genotype]
  def setAnnotation(ann: Annotations)
  def getAnnotation: Annotations
  def setKey(k: K)
  def getKey: K
}

class BgenParsedLine extends ParsedLine[Variant] {
  var gs: Iterable[Genotype] = null
  var variant: Variant = null
  var ann: Annotations = null

  def setGS(gs: Iterable[Genotype]) {
    this.gs = gs
  }
  def getGS: Iterable[Genotype] = gs

  def setAnnotation(ann: Annotations) {
    this.ann = ann
  }
  def getAnnotation: Annotations = ann

  def setKey(k: Variant) {
    variant = k
  }
  def getKey: Variant = variant
}

class PlinkParsedLine extends ParsedLine[Int] {
  var gs: Iterable[Genotype] = null
  var pos: Int = -1
  var ann: Annotations = null

  def setGS(gs: Iterable[Genotype]) {
    this.gs = gs
  }
  def getGS: Iterable[Genotype] = gs

  def setAnnotation(ann: Annotations) {
    this.ann = ann
  }
  def getAnnotation: Annotations = ann

  def setKey(k: Int) {
    pos = k
  }
  def getKey: Int = pos
}

abstract class IndexedBinaryBlockReader[K](job: Configuration, split: FileSplit)
  extends RecordReader[LongWritable, ParsedLine[K]] {

  val LOG: Log = LogFactory.getLog(classOf[IndexedBinaryBlockReader[K]].getName)
  val partitionStart: Long = split.getStart
  var pos: Long = partitionStart
  val end: Long = partitionStart + split.getLength
  val bfis = openFile

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