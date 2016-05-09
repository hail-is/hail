package org.broadinstitute.hail.io

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._
import org.broadinstitute.hail.variant.Genotype
import org.broadinstitute.hail.annotations._
import scala.collection.mutable

class VariantRecord[K] extends Serializable {
  var gs: Iterable[Genotype] = null
  var ann: Annotation = Annotation.empty
  var key: K = _
  var genotypeFlags = mutable.Map.empty[Int, Int]

  def setGS(gs: Iterable[Genotype]) {
    this.gs = gs
  }

  def getGS: Iterable[Genotype] = gs

  def setAnnotation(ann: Annotation) {
    this.ann = ann
  }

  def getAnnotation: Annotation = ann

  def setKey(k: K) {
    this.key = k
  }

  def getKey: K = key

  def getGenotypeFlags = genotypeFlags

  def setGenotypeFlags(idx: Int) {
    this.genotypeFlags(idx) = this.genotypeFlags.getOrElseUpdate(idx, 0) + 1
  }

  def resetGenotypeFlags() {
    this.genotypeFlags = mutable.Map.empty[Int, Int]
  }
}

abstract class IndexedBinaryBlockReader[K](job: Configuration, split: FileSplit)
  extends RecordReader[LongWritable, VariantRecord[K]] {

  val LOG: Log = LogFactory.getLog(classOf[IndexedBinaryBlockReader[K]].getName)
  val partitionStart: Long = split.getStart
  var pos: Long = partitionStart
  val end: Long = partitionStart + split.getLength
  val bfis = openFile()

  def openFile(): HadoopFSDataBinaryReader = {
    val file: Path = split.getPath
    val fs: FileSystem = file.getFileSystem(job)
    new HadoopFSDataBinaryReader(fs.open(file))
  }

  def seekToFirstBlockInSplit(start: Long): Unit

  def createKey(): LongWritable = new LongWritable()

  def createValue(): VariantRecord[K] = new VariantRecord[K]

  def getPos: Long = pos

  def getProgress: Float = {
    if (partitionStart == end)
      0.0f
    else
      Math.min(1.0f, (pos - partitionStart) / (end - partitionStart).toFloat)
  }

  def close() = bfis.close()

}