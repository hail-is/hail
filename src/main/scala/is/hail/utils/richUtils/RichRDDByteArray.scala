package is.hail.utils.richUtils

import is.hail.io.hadoop.{ByteArrayOutputFormat, BytesOnlyWritable}
import is.hail.utils._
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class RichRDDByteArray(val r: RDD[Array[Byte]]) extends AnyVal {
  def saveFromByteArrays(filename: String, tmpDir: String, header: Option[Array[Byte]] = None,
    deleteTmpFiles: Boolean = true, parallelWrite: Boolean = false) {

    val nullWritableClassTag = implicitly[ClassTag[NullWritable]]
    val bytesClassTag = implicitly[ClassTag[BytesOnlyWritable]]
    val hConf = r.sparkContext.hadoopConfiguration

    hConf.delete(filename, recursive = true) // overwriting by default

    val parallelOutputPath =
      if (parallelWrite) {
        filename
      } else
        hConf.getTemporaryFile(tmpDir)

    val rWithHeader = header.map { h =>
      if (r.partitions.length == 0)
        r.sparkContext.parallelize(List(h), 1)
      else if (parallelWrite)
        r.mapPartitions { it => Iterator(h) ++ it }
      else
        r.mapPartitionsWithIndex { case (i, it) =>
          if (i == 0)
            Iterator(h) ++ it
          else
            it
        }
    }.getOrElse(r)

    val rMappedWithHeader = rWithHeader.mapPartitions { iter =>
      val bw = new BytesOnlyWritable()
      iter.map { bb =>
        bw.set(new BytesWritable(bb))
        (NullWritable.get(), bw)
      }
    }

    RDD.rddToPairRDDFunctions(rMappedWithHeader)(nullWritableClassTag, bytesClassTag, null)
      .saveAsHadoopFile[ByteArrayOutputFormat](parallelOutputPath)

    if (!hConf.exists(parallelOutputPath + "/_SUCCESS"))
      fatal("write failed: no success indicator found")

    if (!parallelWrite)
      hConf.copyMerge(parallelOutputPath, filename, deleteTmpFiles, false)
  }
}
