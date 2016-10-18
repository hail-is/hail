package org.broadinstitute.hail.utils.richUtils

import org.apache.hadoop
import org.apache.hadoop.fs.FileStatus
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.driver.HailConfiguration
import org.broadinstitute.hail.sparkextras.ReorderedPartitionsRDD
import org.broadinstitute.hail.utils._

import scala.reflect.ClassTag

class RichRDD[T](val r: RDD[T]) extends AnyVal {
  def countByValueRDD()(implicit tct: ClassTag[T]): RDD[(T, Int)] = r.map((_, 1)).reduceByKey(_ + _)

  def reorderPartitions(oldIndices: Array[Int])(implicit tct: ClassTag[T]): RDD[T] =
    new ReorderedPartitionsRDD[T](r, oldIndices)

  def forall(p: T => Boolean)(implicit tct: ClassTag[T]): Boolean = r.map(p).fold(true)(_ && _)

  def exists(p: T => Boolean)(implicit tct: ClassTag[T]): Boolean = r.map(p).fold(false)(_ || _)

  def writeTable(filename: String, header: Option[String] = None, parallelWrite: Boolean = false, deleteTmpFiles: Boolean = true) {
    val hConf = r.sparkContext.hadoopConfiguration

    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = Option(codecFactory.getCodec(new hadoop.fs.Path(filename)))
    val headerExt = codec.map(_.getDefaultExtension).getOrElse("")

    hConf.delete(filename, recursive = true) // overwriting by default

    val parallelOutputPath =
      if (parallelWrite) {
        filename
      } else
        hConf.getTemporaryFile(HailConfiguration.tmpDir)

    codec match {
      case Some(x) => r.saveAsTextFile(parallelOutputPath, x.getClass)
      case None => r.saveAsTextFile(parallelOutputPath)
    }

    if (!hConf.exists(parallelOutputPath + "/_SUCCESS"))
      fatal("write failed: no success indicator found")

    header.foreach { str =>
      // header will appear first in path sorted order since - comes before 0-9
      val headerPath = parallelOutputPath + "/part--header" + headerExt

      hConf.writeTextFile(headerPath) { s =>
        s.write(str)
        s.write("\n")
      }
    }

    if (!parallelWrite) {
      val filesToMerge = hConf.glob(parallelOutputPath + "/part-*").sortBy(fs => getPartNumber(fs.getPath.getName))

      val (_, dt) = time {
        hConf.copyMerge(filesToMerge, filename, deleteTmpFiles)
      }
      info(s"while writing:\n    $filename\n  merge time: ${ formatTime(dt) }")

      if (deleteTmpFiles)
        hConf.delete(parallelOutputPath, recursive = true)
    }
  }

  def collectOrdered()(implicit tct: ClassTag[T]): Array[T] =
    r.zipWithIndex().collect().sortBy(_._2).map(_._1).toArray
}
