package is.hail.utils.richUtils

import is.hail.io.hadoop.ByteArrayOutputFormat
import is.hail.io.hadoop.BytesOnlyWritable
import is.hail.utils._
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class RichRDDByteArray(val r: RDD[Array[Byte]]) extends AnyVal {
  def saveFromByteArrays(filename: String, tmpDir: String, header: Option[Array[Byte]] = None, deleteTmpFiles: Boolean = true) {
    val nullWritableClassTag = implicitly[ClassTag[NullWritable]]
    val bytesClassTag = implicitly[ClassTag[BytesOnlyWritable]]
    val hConf = r.sparkContext.hadoopConfiguration

    val tmpFileName = hConf.getTemporaryFile(tmpDir)

    val rMapped = r.mapPartitions { iter =>
      val bw = new BytesOnlyWritable()
      iter.map { bb =>
        bw.set(bb)
        (NullWritable.get(), bw)
      }
    }

    rMapped.saveAsHadoopFile[ByteArrayOutputFormat](tmpFileName)

    header.foreach { str =>
      hConf.writeDataFile(tmpFileName + "/header") { s =>
        s.write(str)
      }
    }
    
    hConf.copyMerge(tmpFileName, filename, r.getNumPartitions, deleteTmpFiles, header.isDefined)
  }
}
