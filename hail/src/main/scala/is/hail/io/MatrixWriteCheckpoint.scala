package is.hail.io

import is.hail.io.fs.FS
import is.hail.utils._

import java.io.{Closeable, OutputStreamWriter}

object MatrixWriteConfig {
  def parse(s: String): MatrixWriteConfig = {
    val Array(matrixPath, version, nPartitions) = s.split("\t")
    MatrixWriteConfig(matrixPath, version, nPartitions.toInt)
  }

  def generate(matrixPath: String, nPartitions: Int): MatrixWriteConfig = {
    MatrixWriteConfig(matrixPath, is.hail.HAIL_PRETTY_VERSION, nPartitions)
  }
}

case class MatrixWriteConfig(
  matrixPath: String,
  version: String,
  nPartitions: Int
) {
  def render(): String = s"$matrixPath\t$version\t$nPartitions"
}

object MatrixWriteCheckpoint {

  def parseLine(s: String): (Int, FileWriteMetadata) = {
    val Array(partIndex, path, rowsWrittenStr, bytesWrittenStr) = s.stripLineEnd.split("\t")
    (partIndex.toInt, FileWriteMetadata(path, rowsWrittenStr.toLong, bytesWrittenStr.toLong))
  }

  def read(fs: FS, checkpointFile: String, matrixPath: String, nPartitions: Int): MatrixWriteCheckpoint = {
    val genConfig = MatrixWriteConfig.generate(matrixPath, nPartitions)
    if (fs.exists(checkpointFile)) {
      fs.copy(checkpointFile, checkpointFile + ".bak", deleteSource = true)
      val (config, data) = fs.readLines(checkpointFile + ".bak") { it =>
        val header = it.next().value
        assert(header.startsWith("#"))
        val config = MatrixWriteConfig.parse(header.drop(1))


        if (config != genConfig) {
          fatal(s"invalid checkpoint file:" +
            s"\n  written: path=${ config.matrixPath }, version=${ config.version }, nPartitions=${ config.nPartitions }" +
            s"\n  current: path=${ genConfig.matrixPath }, version=${ genConfig.version }, nPartitions=${ genConfig.nPartitions }")
        }

        val dataLines = it
          .map { line => parseLine(line.value.stripLineEnd) }
          .toMap
        info(s"resuming matrix write from ${ checkpointFile } with ${ dataLines.size }/${ nPartitions } partitions written")
        (config, Array.tabulate(config.nPartitions)(i => dataLines.get(i).orNull))
      }
      new MatrixWriteCheckpoint(fs, checkpointFile, config, data)
    } else {
      info(s"creating new checkpoint at ${ checkpointFile }")
      new MatrixWriteCheckpoint(fs, checkpointFile, genConfig, new Array[FileWriteMetadata](nPartitions))
    }
  }
}

class MatrixWriteCheckpoint(fs: FS, checkpointFile: String, config: MatrixWriteConfig, data: Array[FileWriteMetadata]) extends Closeable {

  private[this] var closed = false
  private[this] val writer = new OutputStreamWriter(fs.create(checkpointFile))
  writer.write(s"#${ config.render() }\n")
  data.iterator
    .filter(_ != null)
    .zipWithIndex
    .foreach { case (fwm, idx) => writeLine(idx, fwm) }
  writer.flush()

  def uncomputedPartitions(): Set[Int] = data.indices.filter(i => data(i) == null).toSet

  def result(): Array[FileWriteMetadata] = {
    data.foreach(fwm => assert(fwm != null, fwm))
    close()
    data
  }

  private[this] def writeLine(partIdx: Int, fwm: FileWriteMetadata): Unit = {
    writer.write(s"$partIdx\t${ fwm.path }\t${ fwm.rowsWritten }\t${ fwm.bytesWritten }\n")
  }

  def append(partIdx: Int, fwm: FileWriteMetadata): Unit = {
    assert(data(partIdx) == null, partIdx)
    data(partIdx) = fwm
    writeLine(partIdx, fwm)
  }

  override def close(): Unit = {
    if (!closed) {
      writer.flush()
      writer.close()
      closed = true
    }
  }
}
