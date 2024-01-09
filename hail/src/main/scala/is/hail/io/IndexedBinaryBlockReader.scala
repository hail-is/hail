package is.hail.io

import is.hail.annotations.RegionValueBuilder
import is.hail.io.fs.{HadoopFS, WrappedSeekableDataInputStream}

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.mapred._

abstract class KeySerializedValueRecord[K] extends Serializable {
  var input: Array[Byte] = _
  var key: K = _

  def setSerializedValue(arr: Array[Byte]) {
    this.input = arr
  }

  def getValue(rvb: RegionValueBuilder, includeGT: Boolean): Unit

  def setKey(k: K) {
    this.key = k
  }

  def getKey: K = key
}

abstract class IndexedBinaryBlockReader[T](job: Configuration, split: FileSplit)
    extends RecordReader[LongWritable, T] {

  val LOG: Log = LogFactory.getLog(classOf[IndexedBinaryBlockReader[T]].getName)
  val partitionStart: Long = split.getStart
  var pos: Long = partitionStart
  val end: Long = partitionStart + split.getLength
  val bfis = openFile()

  def openFile(): HadoopFSDataBinaryReader = {
    val file: Path = split.getPath
    val fs: FileSystem = file.getFileSystem(job)
    val is = fs.open(file)
    new HadoopFSDataBinaryReader(
      new WrappedSeekableDataInputStream(
        HadoopFS.toSeekableInputStream(is)
      )
    )
  }

  def createKey(): LongWritable = new LongWritable()

  def createValue(): T

  def getPos: Long = pos

  def getProgress: Float =
    if (partitionStart == end)
      0.0f
    else
      Math.min(1.0f, (pos - partitionStart) / (end - partitionStart).toFloat)

  def close() = bfis.close()

}
