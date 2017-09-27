package is.hail.utils.richUtils

import is.hail.io.hadoop.ByteArrayOutputFormat
import is.hail.io.hadoop.BytesOnlyWritable
import is.hail.utils._
import org.apache.hadoop.io.NullWritable
import org.apache.spark.rdd.RDD

class RichRDDByteArray(val r: RDD[Array[Byte]]) extends AnyVal {
  def saveFromByteArrays(filename: String, tmpDir: String, header: Option[Array[Byte]] = None,
    deleteTmpFiles: Boolean = true, parallelWrite: Boolean = false) {
    val hConf = r.sparkContext.hadoopConfiguration

    hConf.delete(filename, recursive = true) // overwriting by default

    val parallelOutputPath = if (parallelWrite) filename else hConf.getTemporaryFile(tmpDir)

    val rWithHeader = header.map { h =>
      if (r.partitions.length == 0)
        r.sparkContext.parallelize(List(h), numSlices = 1)
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

    val rMappedWithHeader = rWithHeader.mapPartitions { it =>
      val bw = new BytesOnlyWritable()
      it.map { bb =>
        bw.set(bb)
        (NullWritable.get(), bw)
      }
    }

    rMappedWithHeader.saveAsHadoopFile[ByteArrayOutputFormat](parallelOutputPath)

    if (!hConf.exists(parallelOutputPath + "/_SUCCESS"))
      fatal("write failed: no success indicator found")

    if (!parallelWrite) {
      val expNumPart = if (r.partitions.length == 0) 1 else r.getNumPartitions
      hConf.copyMerge(parallelOutputPath, filename, expNumPart, deleteTmpFiles, hasHeader = false)
    }

  }
}
