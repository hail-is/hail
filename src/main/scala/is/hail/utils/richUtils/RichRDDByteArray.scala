package is.hail.utils.richUtils

import is.hail.io.hadoop.ByteArrayOutputFormat
import is.hail.io.hadoop.BytesOnlyWritable
import is.hail.utils._
import org.apache.hadoop.io.NullWritable
import org.apache.spark.rdd.RDD

class RichRDDByteArray(val r: RDD[Array[Byte]]) extends AnyVal {
  def saveFromByteArrays(filename: String, tmpDir: String, header: Option[Array[Byte]] = None,
    deleteTmpFiles: Boolean = true, exportType: Int = ExportType.CONCATENATED) {
    val hConf = r.sparkContext.hadoopConfiguration

    hConf.delete(filename, recursive = true) // overwriting by default

    val parallelOutputPath =
      if (exportType == ExportType.CONCATENATED)
        hConf.getTemporaryFile(tmpDir)
      else
        filename

    val rWithHeader = header.map { h =>
      if (r.partitions.length == 0 && exportType != ExportType.PARALLEL_SEPARATE_HEADER)
        r.sparkContext.parallelize(List(h), numSlices = 1)
      else {
        exportType match {
          case ExportType.CONCATENATED =>
            r.mapPartitionsWithIndex { case (i, it) =>
              if (i == 0)
                Iterator(h) ++ it
              else
                it
            }
          case ExportType.PARALLEL_SEPARATE_HEADER =>
            r
          case ExportType.PARALLEL_HEADER_IN_SHARD =>
            r.mapPartitions { it => Iterator(h) ++ it }
          case _ => fatal(s"Unknown export type: $exportType")
        }
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


    if (exportType == ExportType.PARALLEL_SEPARATE_HEADER)
      hConf.writeFile(parallelOutputPath + "/header")(out => out.write(header.getOrElse(Array.empty[Byte])))

    if (!hConf.exists(parallelOutputPath + "/_SUCCESS"))
      fatal("write failed: no success indicator found")

    if (exportType == ExportType.CONCATENATED) {
      hConf.copyMerge(parallelOutputPath, filename, rWithHeader.getNumPartitions, deleteTmpFiles, hasHeader = false)
    }
  }
}
