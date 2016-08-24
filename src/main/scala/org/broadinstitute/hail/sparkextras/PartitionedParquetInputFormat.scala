package org.broadinstitute.hail.sparkextras

import org.apache.hadoop.fs._
import org.apache.hadoop.mapred.FileInputFormat
import org.apache.hadoop.mapreduce.{InputSplit, JobContext}
import org.apache.parquet.hadoop.ParquetInputFormat

import scala.collection.JavaConverters._

import java.util.{List => JList}

/**
  * Copied and slightly modified from:
  *   org.apache.spark.sql.execution.datasources.parquet.ParquetInputFormat
  *   version 1.5.0
  *
  * Changed to sort splits by the split index in the file name
  * so that the resulting HadoopRDD has the same partitions as
  * the RDD which was written to disk.
  */
class PartitionedParquetInputFormat[T] extends ParquetInputFormat[T] {

  val partRegex = "part-r-(\\d+)-.*\\.parquet.*".r

  def getPartNumber(fname: String): Int = {
    fname match {
      case partRegex(i) => i.toInt
      case _ => throw new PathIOException("no match")
    }
  }

  override def getSplits(job: JobContext): JList[InputSplit] = {
    val splits: JList[InputSplit] = new java.util.ArrayList[InputSplit]
    val files: JList[FileStatus] = listStatus(job)

    val sorted = files.asScala.toArray.sortBy(fs => getPartNumber(fs.getPath.getName)).toList
    for (file <- sorted) {
      val path: Path = file.getPath
      val length: Long = file.getLen
      if (length != 0) {
        val blkLocations = file match {
          case lfs: LocatedFileStatus => lfs.getBlockLocations
          case _ =>
            val fs: FileSystem = path.getFileSystem(job.getConfiguration)
            fs.getFileBlockLocations(file, 0, length)
        }

        splits.add(makeSplit(path, 0, length, blkLocations(0).getHosts, blkLocations(0).getCachedHosts))
      } else {
        splits.add(makeSplit(path, 0, length, new Array[String](0)))
      }
    }
    job.getConfiguration.setLong(FileInputFormat.NUM_INPUT_FILES, files.size)
    splits
  }
}
