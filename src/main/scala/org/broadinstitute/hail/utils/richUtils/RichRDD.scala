package org.broadinstitute.hail.utils.richUtils

import org.apache.hadoop
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

  def writeTable(filename: String, header: Option[String] = None, deleteTmpFiles: Boolean = true) {
    val hConf = r.sparkContext.hadoopConfiguration
    val tmpFileName = hConf.getTemporaryFile(HailConfiguration.tmpDir)
    val codecFactory = new CompressionCodecFactory(hConf)
    val codec = Option(codecFactory.getCodec(new hadoop.fs.Path(filename)))
    val headerExt = codec.map(_.getDefaultExtension).getOrElse("")

    header.foreach { str =>
      hConf.writeTextFile(tmpFileName + ".header" + headerExt) { s =>
        s.write(str)
        s.write("\n")
      }
    }

    codec match {
      case Some(x) => r.saveAsTextFile(tmpFileName, x.getClass)
      case None => r.saveAsTextFile(tmpFileName)
    }

    val filesToMerge = header match {
      case Some(_) => Array(tmpFileName + ".header" + headerExt, tmpFileName + "/part-*")
      case None => Array(tmpFileName + "/part-*")
    }

    hConf.delete(filename, recursive = true) // overwriting by default

    val (_, dt) = time {
      hConf.copyMerge(filesToMerge, filename, deleteTmpFiles)
    }
    info(s"while writing:\n    $filename\n  merge time: ${ formatTime(dt) }")

    if (deleteTmpFiles) {
      hConf.delete(tmpFileName + ".header" + headerExt, recursive = false)
      hConf.delete(tmpFileName, recursive = true)
    }
  }
}
