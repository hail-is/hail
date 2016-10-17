package org.broadinstitute.hail.utils.richUtils

import org.apache.hadoop.fs.{FileStatus, PathIOException}
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.driver.HailConfiguration
import org.broadinstitute.hail.io.hadoop.{ByteArrayOutputFormat, BytesOnlyWritable}
import org.broadinstitute.hail.utils._

import scala.reflect.ClassTag

class RichRDDByteArray(val r: RDD[Array[Byte]]) extends AnyVal {
  def saveFromByteArrays(filename: String, header: Option[Array[Byte]] = None, deleteTmpFiles: Boolean = true) {
    def getPartNumber(fileStatus: FileStatus): Int = {
      val partRegex = """.*/?part-(\d+).*""".r

      val fname = fileStatus.getPath.getName
      fname match {
        case partRegex(i) => i.toInt
        case _ => throw new PathIOException(s"invalid parquet file `$fname'")
      }
    }

    val nullWritableClassTag = implicitly[ClassTag[NullWritable]]
    val bytesClassTag = implicitly[ClassTag[BytesOnlyWritable]]
    val hConf = r.sparkContext.hadoopConfiguration

    val tmpFileName = hConf.getTemporaryFile(HailConfiguration.tmpDir)

    header.foreach { str =>
      hConf.writeDataFile(tmpFileName + ".header") { s =>
        s.write(str)
      }
    }

    val filesToMerge = header match {
      case Some(_) => Array((tmpFileName + ".header", null), (tmpFileName + "/part-*", getPartNumber _))
      case None => Array((tmpFileName + "/part-*", getPartNumber _))
    }

    val rMapped = r.mapPartitions { iter =>
      val bw = new BytesOnlyWritable()
      iter.map { bb =>
        bw.set(new BytesWritable(bb))
        (NullWritable.get(), bw)
      }
    }

    RDD.rddToPairRDDFunctions(rMapped)(nullWritableClassTag, bytesClassTag, null)
      .saveAsHadoopFile[ByteArrayOutputFormat](tmpFileName)

    if (!hConf.exists(tmpFileName + "/_SUCCESS"))
      fatal("write failed: no success indicator found")

    hConf.delete(filename, recursive = true) // overwriting by default

    val (_, dt) = time {
      hConf.copyMerge(filesToMerge, filename, deleteTmpFiles)
    }
    println("merge time: " + formatTime(dt))

    if (deleteTmpFiles) {
      hConf.delete(tmpFileName + ".header", recursive = false)
      hConf.delete(tmpFileName, recursive = true)
    }
  }
}
